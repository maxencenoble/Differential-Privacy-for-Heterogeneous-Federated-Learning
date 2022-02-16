import torch
import os
import h5py
from flearn.users.user_scaffold import UserSCAFFOLD
from flearn.servers.server_base import Server
from utils.model_utils import read_data, read_user_data, read_data_cross_validation
from scipy.stats import rayleigh
import numpy as np


# Implementation for SCAFFOLD Server
class SCAFFOLD(Server):
    def __init__(self, dataset, algorithm, model, nb_users, nb_samples, user_ratio, sample_ratio, L,
                 local_learning_rate, max_norm, num_glob_iters, local_updates, users_per_round, similarity, noise,
                 times, dp, sigma_gaussian, alpha, beta, number, dim_pca, use_cuda, warm_start, k_fold=None,
                 nb_fold=None):

        if similarity is None:
            similarity = (alpha, beta)

        if alpha < 0 and beta < 0:
            similarity = "iid"

        super().__init__(dataset, algorithm, model[0], nb_users, nb_samples, user_ratio, sample_ratio, L, max_norm,
                         num_glob_iters, local_updates, users_per_round, similarity, noise, times, dp, sigma_gaussian,
                         number, model[1], use_cuda)
        self.control_norms = []
        self.warm_start = warm_start

        local_epochs = max(round(self.local_updates * sample_ratio),1)

        # definition of the local learning rate
        self.local_learning_rate = local_learning_rate / (local_epochs * self.global_learning_rate)

        if model[1][-3:] != "PCA":
            dim_pca = None

        # Initialize data for all  users
        if k_fold is None:
            data = read_data(dataset, self.number, str(self.similarity), dim_pca)
        else:
            # Cross Validation
            data = read_data_cross_validation(dataset, self.number, str(self.similarity), k_fold, nb_fold, dim_pca)

        total_users = len(data[0])
        for i in range(total_users):
            id, train, test = read_user_data(i, data, dataset)

            user = UserSCAFFOLD(id, train, test, model, sample_ratio, self.local_learning_rate, L,
                                local_updates, dp, times, use_cuda)
            self.users.append(user)
            self.total_train_samples += user.train_samples

        if self.noise:
            self.communication_thresh = rayleigh.ppf(1 - users_per_round / total_users)  # h_min

        print("Number of users / total users:", users_per_round, " / ", total_users)

        self.server_controls = [torch.zeros_like(p.data) for p in self.model.parameters() if p.requires_grad]
        self.seen_users_controls = []

        print("Finished creating SCAFFOLD server.")

    def train(self):
        loss = []
        for glob_iter in range(self.num_glob_iters):
            print("-------------Round number: ", glob_iter, " -------------")
            # loss_ = 0

            # Each user gets the global parameters and sets their controls
            self.send_parameters(glob_iter)

            # WARM START : wise initialisation depending on x_0
            if self.warm_start and glob_iter == 0:
                self.set_controls_all_users()

            # Evaluate model at each iteration
            self.evaluate()

            # Users are selected
            if self.noise:
                self.selected_users = self.select_transmitting_users()
                print(f"Transmitting {len(self.selected_users)} users")
            else:
                self.selected_users = self.select_users(glob_iter, self.users_per_round)

            # Local updates
            for user in self.selected_users:

                seen = (user.user_id in self.seen_users_controls)

                if self.dp == "None":
                    user.train_no_dp(glob_iter, self.user_ratio, self.warm_start, seen)
                else:
                    user.train_dp(self.sigma_g, glob_iter, self.user_ratio, self.max_norm,
                                  self.warm_start, seen)

                self.seen_users_controls.append(user.user_id)

                user.drop_lr()

            # Aggregation

            self.aggregate_parameters()
            self.get_max_norm()

            if self.noise:
                self.apply_channel_effect()

        self.save_results()
        self.save_norms()
        self.save_model()

    def send_parameters(self, glob_iter):
        """Users setting their parameters and controls from the server."""
        assert (self.users is not None and len(self.users) > 0)
        for user in self.users:
            user.set_parameters(self.model)

            # for the first 4/self.user_ratio rounds : warm start-strategy on c_i (c remains zero for users)
            if (not self.warm_start) or glob_iter >= round(4 / self.user_ratio):
                for control, new_control in zip(user.server_controls, self.server_controls):
                    control.data = new_control.data

    def set_controls_all_users(self):
        """Setting the initial control variables for all users."""
        assert (self.users is not None and len(self.users) > 0)
        for user in self.users:
            self.set_controls(user)
            print("C_io done :", user.user_id)

    def set_controls(self, user):
        """Setting the initial control variables for user."""
        if self.dp == "None":
            user.set_first_controls_no_dp()
        else:
            user.set_first_controls_dp(self.sigma_g, self.max_norm)

    def aggregate_parameters(self):
        """Aggregation update of the server model."""
        assert (self.users is not None and len(self.users) > 0)
        total_train = 0
        for user in self.selected_users:
            total_train += user.train_samples
        for user in self.selected_users:
            self.add_parameters(user, user.train_samples / total_train)

    def add_parameters(self, user, ratio):
        """Adding to the server model the contribution term from user."""
        # num_of_selected_users = len(self.selected_users)
        num_of_users = len(self.users)
        for param, control, del_control, del_model in zip(self.model.parameters(), self.server_controls,
                                                          user.delta_controls, user.delta_model):
            param.data = param.data + self.global_learning_rate * del_model.data * ratio
            control.data = control.data + del_control.data / num_of_users

            # below : same sample size for all users
            # param.data = param.data + self.global_learning_rate * del_model.data / num_of_selected_users

    def get_max_norm(self):
        """Getting the maximum ||x_user^t+1 -x_server^t|| & ||c_user^t+1 -c_server^t|| over the users"""
        param_norms = []
        control_norms = []
        for user in self.selected_users:
            param_norm, control_norm = user.get_params_norm()
            param_norms.append(param_norm)
            control_norms.append(control_norm)
        self.param_norms.append(max(param_norms))
        self.control_norms.append((max(control_norms)))

    def apply_channel_effect(self, sigma=1, power_control=2500):
        num_of_selected_users = len(self.selected_users)
        alpha_t_params = power_control / self.param_norms[-1] ** 2
        alpha_t_controls = 4e4 * power_control / self.control_norms[-1] ** 2
        for param, control in zip(self.model.parameters(), self.server_controls):
            param.data = param.data + sigma / (
                    alpha_t_params ** 0.5 * num_of_selected_users * self.communication_thresh) * torch.randn(
                param.data.size())
            control.data = control.data + sigma / (
                    alpha_t_controls ** 0.5 * num_of_selected_users * self.communication_thresh) * torch.randn(
                control.data.size())
