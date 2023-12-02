import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
from sklearn.decomposition import PCA


class GNN_analysis():
    def __init__(self):
        self.df_gnnhp = df = pd.DataFrame(
            columns = ['params', 'exist_val', 'time', 'MAE', 'RMSE', 'L1'])  # estudo hiperparâmetros GNN
        self.get_data()

    def get_data(self):
        for exist_val in [0.30, 0.90]:
            for exp in range(3):
                path_baseline = f'GRAPE/uci/mdi_results/results/gnn_mdi_v4/concrete/{exist_val}/{exp}/0/result.pkl'
                with open(path_baseline, 'rb') as file:
                    experiment = pickle.load(file)

                mae = np.mean(np.abs(experiment['outputs']['final_pred_test'] - experiment['outputs']['label_test']))
                rmse = min(experiment['curves']['test_rmse'])
                l1 = min(experiment['curves']['test_l1'])
                self.df_gnnhp.loc[len(self.df_gnnhp.index)] = [experiment['outputs']['params'],
                                                               exist_val,
                                                               experiment['outputs']['time'],
                                                               mae, rmse, l1]

        for version in ['v1', 'v2', 'v3']:
            for exist_val in [0.30, 0.90]:
                for exp in range(8):
                    path_baseline = f'GRAPE/uci/mdi_results/results/gnn_mdi_{version}/concrete/{exist_val}/0/{exp}/result.pkl'
                    with open(path_baseline, 'rb') as file:
                        experiment = pickle.load(file)

                    mae = np.mean(
                        np.abs(experiment['outputs']['final_pred_test'] - experiment['outputs']['label_test']))
                    rmse = min(experiment['curves']['test_rmse'])
                    l1 = min(experiment['curves']['test_l1'])
                    self.df_gnnhp.loc[len(self.df_gnnhp.index)] = [experiment['outputs']['params'],
                                                                   exist_val,
                                                                   experiment['outputs']['time'],
                                                                   mae, rmse, l1]

        def transform_column(column):
            parts = column.split('_')
            transformed_parts = [part.capitalize() for part in parts[1:]]
            return '_'.join(transformed_parts)

        self.df_gnnhp['params'] = self.df_gnnhp['params'].apply(transform_column)
        self.df_gnnhp['miss_val'] = round(1 - self.df_gnnhp['exist_val'], 1)

    def plot_params_gnn(self, metric = 'RMSE', plot_time = True):
        if plot_time:
            fig, axs =(plt.subplots(2, 1, figsize = (15, 12)))
            ax = axs[0]
        else:
            fig, ax =(plt.subplots(1, 1, figsize = (15, 6)))

        sns.barplot(data = self.df_gnnhp, x = 'miss_val', y = metric, hue = 'params', ax = ax)
        ax.set_title(metric, fontsize = 14)
        ax.set_xlabel('% Missing values ', fontsize = 12)
        ax.set_ylabel(metric)

        x_min = 0.03
        x_max = 0.50
        for miss in [0.10, 0.70]:
            df_metric = self.df_gnnhp[self.df_gnnhp['miss_val'] == miss]
            min_metric = min(df_metric.groupby(by = ['params'])[metric].mean())
            ax.axhline(xmin = x_min - 0.03, xmax = x_max + 0.03, y = min_metric, lw = 4, color = 'white')

            ax.axhline(xmin = x_min - 0.03, xmax = x_max + 0.03, y = min_metric, lw = 4, color = 'white')
            x_min += 1 / 2
            x_max += 1 / 2

        ax.axhline(xmin = 0, xmax = 0, y = 0, lw = 4, color = 'white', label = metric + ' Mínimo')
        ax.legend(title = 'Parameters -> FuncAtiv_Optmizer_LR')

        if plot_time:
            ax = axs[1]
            ax = sns.barplot(data = self.df_gnnhp, x = 'miss_val', y = 'time', hue = 'params', ax = ax)
            ax.set_title("Tempo de execução", fontsize = 14)
            ax.set_xlabel('% Missing values ', fontsize = 12)
            ax.set_ylabel("Tempo de execução", fontsize = 12)

            for miss in [0.10, 0.70]:
                df_metric = self.df_gnnhp[self.df_gnnhp['miss_val'] == miss]
                min_metric = min(df_metric.groupby(by = ['params'])[metric].mean())

            ax.legend(title = 'Parameters -> FuncAtiv_Optmizer_LR')

        plt.tight_layout()
        plt.show()



class EarlyStop:
    def __init__(self, patience = 3, delta = 0.5, mode = 'min'):
        self.patience = patience  # Number of epochs with no improvement after which training will be stopped
        self.delta = delta  # Minimum change in the monitored quantity to qualify as an improvement
        self.mode = mode.lower()  # 'min' or 'max' - whether the metric should be minimized or maximized
        self.counter = 0  # Counter to track epochs with no improvement
        self.best_metric = np.Inf if self.mode == 'min' else -np.Inf  # Initialize best metric
        self.stop = False  # Indicator to stop training

    def __call__(self, current_metric):
        if self.mode == 'min':
            if current_metric < self.best_metric - self.delta:
                self.best_metric = current_metric
                self.counter = 0
            else:
                self.counter += 1
                if self.counter >= self.patience:
                    self.stop = True
        else:
            if current_metric > self.best_metric + self.delta:
                self.best_metric = current_metric
                self.counter = 0
            else:
                self.counter += 1
                if self.counter >= self.patience:
                    self.stop = True

    def should_stop(self):
        return self.stop

    # TODO: implement EarlyStop -> didn't reach stop criteria on the current project
    # delta = np.e-3
    # early_rmse = EarlyStop(patience = 3, delta = delta, mode = 'min')
    # early_l1 = EarlyStop(patience = 3, delta = delta, mode = 'min')
    # early_train = EarlyStop(patience = 3, delta = delta, mode = 'min')
    #
    # early_dict = dict_grape['concrete'][0.3]['curves']
    #
    # early_cond = (early_rmse.should_stop() and early_l1.should_stop() and early_train.should_stop())
    # i = 0
    # while not early_cond:
    #     early_rmse(early_dict['test_rmse'][i])
    #     early_l1(early_dict['test_l1'][i])
    #     early_train(early_dict['train_loss'][i])
    #
    #     # early_cond = (early_rmse.should_stop() or early_l1.should_stop() or early_train.should_stop())
    #     early_cond = (early_rmse.should_stop())
    #     i += 1
    #     if(i == len(dict_grape['concrete'][0.3]['curves']['test_rmse']) - 1):
    #         print(f"Não foi alcançado critério de parada para {i+1} epochs")
    #         break





def view_pca(data_dict):
    def apply_pca(data):
        df = data.iloc[:,:-1]
        pca = PCA(n_components=3)
        pca_result = pca.fit(df)
        return pca_result.transform(df), pca_result.explained_variance_

    pca_df1, var_df1 = apply_pca(data_dict['concrete'])
    pca_df2, var_df2 = apply_pca(data_dict['energy'])
    pca_df3, var_df3 = apply_pca(data_dict['power'])

    fig = plt.figure(figsize=(20, 12))
    ax1 = fig.add_subplot(121, projection='3d')


    ax1.scatter(pca_df1[:, 1], pca_df1[:, 0], pca_df1[:, 2], color = 'green', label=f'Concrete: ({100*sum(var_df1):.0f}%)')
    ax1.scatter(pca_df2[:, 1], pca_df2[:, 0], pca_df2[:, 2], color = 'gray', label=f'Energy:   ({100*sum(var_df2):.0f}%)')
    ax1.scatter(pca_df3[:, 1], pca_df3[:, 0], pca_df3[:, 2], color = 'red', label=f'Power:    ({100*sum(var_df3):.0f}%)')

    ax1.set_xlabel('PC1')
    ax1.set_ylabel('PC2')
    ax1.set_zlabel('PC3')
    ax1.legend()

    ax2 = fig.add_subplot(122, projection='3d')

    ax2.scatter(pca_df1[:, 1], pca_df1[:, 0], pca_df1[:, 2], color = 'green', label=f'Concrete: ({100*sum(var_df1):.0f}%)')
    ax2.scatter(pca_df2[:, 1], pca_df2[:, 0], pca_df2[:, 2], color = 'gray', label=f'Energy:   ({100*sum(var_df2):.0f}%)')
    ax2.scatter(pca_df3[:, 1], pca_df3[:, 0], pca_df3[:, 2], color = 'red', label=f'Power:    ({100*sum(var_df3):.0f}%)')
    ax2.view_init(elev=25, azim=185)

    ax2.set_xlabel('PC1')
    ax2.set_ylabel('PC2')
    ax2.set_zlabel('PC3')
    ax2.legend()

    plt.title('PCA Results of Three Dataframes')
    plt.show()

def plot_learning(dict_metadata, seed, data, exist_val):

    data_gnn = dict_metadata[seed][data]['gnn_mdi'][exist_val]['curves']

    figure, axs = plt.subplots(1, 2, figsize = (15, 6))

    ax = axs[0]
    y = data_gnn['train_loss']
    x = np.arange(0, len(y), 1)
    ax.plot(x, y, lw = .75, color = 'black', alpha = 0.25)
    ax.set_xlabel('Epochs', fontsize = 12)
    ax.set_ylabel('Train loss (log)', fontsize = 12)
    ax.set_yscale('log')

    min_index = np.argmin(y)
    min_value = y[min_index]
    ax.axhline(xmin = 0, xmax = min_index / len(y) - 0.05, y = min_value, ls = '--', lw = 2, color = 'red')
    sns.scatterplot(x = [min_index], y = [min_value], s = 100, color = 'r',
                    label = f'Mínimo:   {min_value:.4f} (epoch = {min_index})', ax = ax)

    ax.axhline(xmin = 0, xmax = 1, y = y[-1], ls = '-.', lw = 2, color = 'blue')
    sns.scatterplot(x = [len(y)], y = [y[-1]], s = 75, color = 'b',
                    label = f'Fim treino: {y[-1]:.4f} (epoch = {len(y)})', ax = ax)
    ax.legend(title = 'Train loss', loc = 'upper right')

    ax = axs[1]
    y = data_gnn['test_rmse']
    x = np.arange(0, len(y), 1)
    ax.plot(x, y, lw = .75, color = 'black', alpha = 0.25)
    ax.set_xlabel('Epochs', fontsize = 12)
    ax.set_ylabel('RMSE (log)', fontsize = 12)
    ax.set_yscale('log')

    min_index = np.argmin(y)
    min_value = y[min_index]
    ax.axhline(xmin = 0, xmax = min_index / len(y) + 0.03, y = min_value, ls = '--', lw = 2, color = 'red')
    sns.scatterplot(x = [min_index], y = [min_value], s = 100, color = 'r',
                    label = f'Mínimo:   {min_value:.4f} (epoch = {min_index})', ax = ax)

    ax.axhline(xmin = 0, xmax = 1, y = y[-1], ls = '-.', lw = 2, color = 'blue')
    sns.scatterplot(x = [len(y)], y = [y[-1]], s = 75, color = 'b',
                    label = f'Fim treino: {y[-1]:.4f} (epoch = {len(y)})', ax = ax)
    ax.legend(title = 'RMSE', loc = 'upper right')

    plt.show()