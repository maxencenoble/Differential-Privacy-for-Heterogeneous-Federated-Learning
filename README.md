# **Differential Privacy for Heterogeneous Federated Learning : Utility \& Privacy tradeoffs**

This repository is the official code for
paper [Differential Privacy for Heterogeneous Federated Learning](https://arxiv.org/abs/2111.09278), AISTATS 2022.

In our paper, we propose an algorithm DP-SCAFFOLD(-warm), which is a new version of the so-called SCAFFOLD algorithm (
warm version : wise initialisation of parameters), to tackle heterogeneity issues under mathematical privacy constraints
known as "Differential Privacy" (DP) in a federated learning framework. Using fine results of DP theory, we have
succeeded in establishing both privacy and utility guarantees, which show the superiority of DP-SCAFFOLD over the naive
algorithm DP-FedAvg. We here provide numerical experiments that confirm our analysis and prove the significance of gains
of DP-SCAFFOLD especially when the number of local updates or the level of heterogeneity between users grows.

Two datasets are studied in the paper:

- Real-world data,``Femnist``, (an extended version of EMNIST dataset for federated learning), which you see the
  Accuracy growing with the number of communication rounds (50 local updates first and then 100 local updates) under the
  same DP framework for the algorithms.

![image_femnist](pictures/femnist_accuracy_k_10-1.png)
![image_femnist](pictures/femnist_accuracy_k_20-1.png)

- Synthetic data, ``Logistic``, for logistic regression models, which you see the train loss decreasing with the number
  of communication rounds (50 local updates first and then 100 local updates),under the same DP framework for the
  algorithms.

![image_logistic](pictures/logistic_loss_k_10-1.png)
![image_logistic](pictures/logistic_loss_k_20-1.png)

These results were obtained using parameters available [here](pictures/_parameters.txt). Significant results are also
available for both of these datasets for logistic regression models in the paper.

Remark that the code may be run on other real-world datasets: `CIFAR_10` (**but not stable on GPU...**) and `Mnist`.

# Structure of the code

- `main.py`: five global options (mutually exclusive) are available, once the parameters are given.
    - `generate`: to generate data, introduce heterogeneity, split data between users for federated learning and
      preprocess data. Remark there is the option `generate_pca` which enables to reduce the dimension of the input data
      by applying a non-private PCA on the whole dataset (combined with `dim_pca`, not available on `Logistic`
      dataset).
    - `optimum` (after `generate`): to run the training phase of a model with "unsplitted" data (that is centralized
      dataset) and save the empirical model with the lowest train loss to properly compute the train log-loss gap.
    - `tuning` (after `optimum`): to run 5-fold cross-validation with the selected model, given multiple values of local
      learning rate (fill the variable `local_learning_rate_list` in `main.py`). The results are stored in the
      folder `results_tuning`.
    - `simulation` (after `tuning`): to run several simulations of federated learning once the best local learning rate
      is determined and save the results (accuracy, loss...). This option calls `simulate.py`.
    - `plot` (after `simulation`): to plot visuals.
- `get_epsilon_bound.py`: to obtain a DP epsilon bound from the input parameters (relying on RDP upper bounds).

**First example usage: synthetic data**

``` bash
# 1. Obtain the privacy guarantee given the parameters (fill by hand in get_epsilon_bound.py)
python get_epsilon_bound.py
# 2. Generate data
python main.py --generate 1 --dataset Logistic --alpha 0.0 --beta 0.0 --nb_users 100 --nb_samples 5000 --dim_input 40 --dim_output 10
# 3. Choose a ML model (here mclr) and train it in a centralised setting
python main.py --optimum 1 --dataset Logistic --model mclr --alpha 0.0 --beta 0.0 --dim_input 40 --dim_output 10
# 4. Tune the local_learning_rate parameter in the FL setting, either DP or not (first complete by hand the variable in main.y)
python main.py --tuning 1 --dataset Logistic --model mclr --alpha 0.0 --beta 0.0 --nb_users 100 --user_ratio 0.2 --nb_samples 5000 --sample_ratio 0.2 --dim_input 40 --dim_output 10 --algorithm FedAvg --times 3 --dp Gaussian --sigma_gaussian 50. --num_glob_iters 200 --local_updates 10
# 5. Choose the accurate local_learning_rate and run the training phase
python main.py --tuning 1 --dataset Logistic --model mclr --alpha 0.0 --beta 0.0 --nb_users 100 --user_ratio 0.2 --nb_samples 5000 --sample_ratio 0.2 --dim_input 40 --dim_output 10 --algorithm FedAvg --times 3 --dp Gaussian --sigma_gaussian 50. --num_glob_iters 200 --local_updates 10 --local_learning_rate 0.01
# 6 Plot visuals (assuming you already ran the non DP experiment)
python main.py --plot 1 --dataset Logistic --model mclr --alpha 0.0 --beta 0.0 --sigma_gaussian 50. --local_updates 10 --user_ratio 0.2 --sample_ratio 0.2
```

**Second example usage: real-world data**
``` bash
# 1. Obtain the privacy guarantee given the parameters (fill by hand in get_epsilon_bound.py)
python get_epsilon_bound.py
# 2. Generate data
python main.py --generate_pca 1 --dataset Femnist --similarity 0.0 --nb_users 40 --nb_samples 2500 --dim_pca 60
# 3. Choose a ML model (here NN1_PCA) and train it in a centralised setting
python main.py --optimum 1 --dataset Femnist --model NN1_PCA --similarity 0.0 --dim_pca 60
# 4. Tune the local_learning_rate parameter in the FL setting, either DP or not (first complete by hand the variable in main.y)
python main.py --tuning 1 --dataset Femnist --model NN1_PCA --similarity 0.0 --nb_users 40 --user_ratio 0.2 --nb_samples 2500 --sample_ratio 0.2 --dim_pca 60 --algorithm FedAvg --times 3 --dp Gaussian --sigma_gaussian 50. --num_glob_iters 200 --local_updates 10
# 5. Choose the accurate local_learning_rate and run the training phase
python main.py --tuning 1 --dataset Femnist --model NN1_PCA --similarity 0.0 --nb_users 40 --user_ratio 0.2 --nb_samples 2500 --sample_ratio 0.2 --dim_pca 60 --algorithm FedAvg --times 3 --dp Gaussian --sigma_gaussian 50. --num_glob_iters 200 --local_updates 10 --local_learning_rate 0.01
# 6 Plot visuals (assuming you already ran the non DP experiment)
python main.py --plot 1 --dataset Femnist --model NN1_PCA --similarity 0.0 --sigma_gaussian 50. --local_updates 10 --user_ratio 0.2 --sample_ratio 0.2
```

## ./data

Contains generators of synthetic (`Logistic`) and real-world (`Femnist`, `Mnist`, `CIFAR_10`) data, generated from the
local file `data_generator.py`, designed for a federated learning framework under some similarity parameter. Each folder
contains a folder `data` where the generated data (`train` and `test`) is stored.

## ./flearn

- [differential_privacy](flearn/differential_privacy) : contains code to apply Gaussian mechanism (designed to add
  differential privacy to mini-batch stochastic gradients).
- [optimizers](flearn/optimizers) : contains the optimization framework for each algorithm (adaptation of stochastic
  gradient descent).
- [servers](flearn/servers) : contains the super class `Server` (in `server_base.py`) which is adapted to FedAvg and
  SCAFFOLD, to perform training wrt to the server parameters.
- [trainmodel](flearn/trainmodel) : contains the learning model structures (`MclrLogistic` for logistic
  regression, `NN1` for neural network with one hidden layer, `NN1_PCA` with PCA on the input with the previous
  model, `CNN`)
- [users](flearn/users) : contains the super class `User` (in `user_base.py`) which is adapted to FedAvg and SCAFFOLD to
  perform training wrt to any user parameters.

## ./models

Stores the latest models over the training phase of federated learning.

## ./results

Stores several metrics of convergence for each simulation, each similarity/privacy setting and each algorithm.

Metrics (evaluated at each round of communication):

- test accuracy over all users,
- train loss over all users,
- highest norm of parameter difference (server/user) over all selected users,
- train gradient dissimilarity over all users.

# Software requirements

- To download the dependencies: **pip install -r requirements.txt**.

# Citation

If you use this code, please cite the following (BibTex format):

``` bash
@inproceedings{Noble2022dpscaffold,
  title        = {Differentially Private Federated Learning on Heterogeneous Data},
  author       = {Noble, Maxence and Bellet, Aur{\'e}lien and Dieuleveut Aymeric},
  booktitle    = {Artificial intelligence and statistics},
  year         = {2022},
  organization = {PMLR}
}
```

# References

- Paper : https://arxiv.org/abs/2111.09278
- Main inspiration of the
  code: https://github.com/ramshi236/Accelerated-Federated-Learning-Over-MAC-in-Heterogeneous-Networks
- Per-sample gradients: https://github.com/cybertronai/autograd-hacks/blob/master/autograd_hacks.py
- SCAFFOLD & FedAvg paper: https://arxiv.org/abs/1910.06378
- Generation of Logistic data and introduction of heterogeneity: https://arxiv.org/abs/1812.06127
- Creation of dissimilarity for FEMNIST data: https://arxiv.org/abs/1909.06335
