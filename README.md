reference GNN: https://snap.stanford.edu/grape/

# Initial Study on GNN - Hyperparameters and Convergence

## Epochs
Number of epochs: Initial study 20000
It is observed, through the Train loss, that the GNN was still learning even with 20000 epochs - variation around the train loss indicates the need to optimize the network architecture (number of layers, etc.) and/or parameters.
With many missing data (30% existing), the minimum RMSE is reached early and stabilizes at a higher value.
With few missing data (90% existing), the minimum RMSE is reached almost at the end of training.
None of the cases reached Earlystop with tol = 1e-3 and patience = 5.


![image](https://github.com/arthurhirez/AM_IA/assets/109704516/8d22c8b7-bf4a-4a08-bf5c-f57d97d0bbdf)

Conclusion: For time criteria, experiments will have 12000 epochs.


## Hyperparameters

Activation function: [Relu, Prelu]
Optimizer: [Adam, RMSprop]
Learning rate: [0.01*, 0.001, 0.005]

Study conducted for extreme cases: 10% and 70% missing values
Metrics: [MAE, RMSE]

Prelu_Adam_0.005 achieved better results for few missing data (10%), but in general, all (except LR=0.01 and Prelu_Rmsprop_0.005) performed similarly with a lot of missing data (70%) in both metrics.

For runtime criteria, the following were adopted:
Activation function: [Relu]
Optimizer: [Adam]
Learning rate: [0.005]
With a higher learning rate, the number of epochs is reduced from 20000 to 12000.
![image](https://github.com/arthurhirez/AM_IA/assets/109704516/f751fb2c-f967-4247-aeae-8989f38db702)
![image](https://github.com/arthurhirez/AM_IA/assets/109704516/d6e0dbce-c42a-4bd3-8006-0ea9648cca2f)


# Dataset Visualization

**Concrete:**
Univariate analysis with ~1000 observations

**Power:**
Univariate analysis with ~10000 observations

**Energy:**
Multivariate analysis (2 labels in the original, adopted 1 label in this work)

**Bank:**
Univariate analysis, unique with the presence of categorical variables (for exploration purposes, since the network is not programmed to handle imputation of categorical data, but we would like to see how it performs in this case)

Attempt to visualize the similarity of datasets via PCA with 3 components. It explains the UCI datasets well and only a small portion (~50%) of the bank dataset, probably due to the large number of categorical variables. However, it is possible to see that the chosen datasets provide relative diversity for testing.

Comparisons will be mainly based on the Concrete and Power datasets, as they have similar characteristics (univariate analysis and only numerical variables in the original dataset), with the main difference being the size of the dataset.

![image](https://github.com/arthurhirez/AM_IA/assets/109704516/1320ad18-66c8-49ed-bd32-3efbcfcd3095)


# Imputation Methods Analysis

The following methods were used (without optimization):
- Mean
- KNN
- MICE
- SVD
- Spectral (explain -> fancyimputer)

Compared with GRAPE (GNN without optimization)

Best performers: GNN - MICE - KNN

RMSE: GRAPE always better, MICE is baseline in almost all cases

MAE: Large dataset (power) with 50% mean better than KNN, MICE remains good, KNN -> improve cluster number guess?

![image](https://github.com/arthurhirez/AM_IA/assets/109704516/3b99e8fb-d7df-48a5-addb-db91ce682135)

![image](https://github.com/arthurhirez/AM_IA/assets/109704516/a62a7dca-0bf4-44e2-89e1-3fedd3c373bc)

![image](https://github.com/arthurhirez/AM_IA/assets/109704516/8ddb78e8-6d34-45d0-bd50-52daf1632aac)


# Analysis of the Imputation Influence on Prediction

Regression algorithms used (without hyperparameter optimization):

- Linear Regression
- Elastic Net CV
- Polynomial Features
- Lasso (Note: Applies Lasso to Polynomial, with alpha estimated using CV)
- Decision Tree
- Random Forest
- Gradient Boosting
- KNeighbors Uniform
- KNeighbors Distance


In the visualizations below:
GRAPE did not perform well; the training stopped too early.

Gradient boosting and Random forest generalized better.
Interestingly, the best result (RMSE) was obtained by inputting the values with the mean of the data in some cases.

Linear/polynomial regression performed poorly with data that had very poor imputation (SVD and spectral).

The smallest error was about ~300% of the error for the "Concrete" data with 30% missing values.

It is evident, comparing the "Concrete" data with "Power," that the greater availability of data in the Power dataset resulted in lower errors for most regression algorithms.

![image](https://github.com/arthurhirez/AM_IA/assets/109704516/2197396f-2397-4988-a427-35f96edc3aa1)

![image](https://github.com/arthurhirez/AM_IA/assets/109704516/ef7d9485-2fd7-4b95-9c0d-9cd34045a91c)

![image](https://github.com/arthurhirez/AM_IA/assets/109704516/fd1b234d-045a-46c4-9067-5f590b9ad1f6)

![image](https://github.com/arthurhirez/AM_IA/assets/109704516/e6d7c3a4-5547-4ff8-a0c3-ae8f8c3a7059)


Conclusions
- Common situation: missing feature values in datasets, on different scales.
- The use of GNN showed superior results in the observed metrics compared to established algorithms in the literature for imputation.
- Training time, complexity, and dependence on a considerable amount of data hinder the adoption of GNN.
- The use of pre-trained models and hyperparameter optimization can be effective in overcoming adoption challenges.
- GNN is versatile, being used for both imputation and regression - there is only the need to train a single model for both tasks.
