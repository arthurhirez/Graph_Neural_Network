import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle

from sklearn.preprocessing import MinMaxScaler


class Experiment_Input():

    def __init__(self,
                 seed_list = None, data_list = None, input_list = None,
                 value_list = None):

        self.seeds = [0] if seed_list is None else seed_list
        self.data_base = ['concrete'] if data_list is None else data_list
        self.methods_base = ['gnn_mdi', 'knn'] if input_list is None else input_list
        self.exist_val_percent = [0.30, 0.50, 0.70, 0.90] if value_list is None else value_list

        self.dict_metadata = {}  # Metadados dos algoritmos
        self.data_dict = {}  # Dataframe original -> data_dict[data]
        self.missing_dict = {}  # Valores faltantes  -> missing_dict[seed][data][exist_val]
        self.filled_dict = {}  # Valores imputados  -> filled_dict[seed][data][method][exist_val]
        self.predict_dict = {}  # Valores preditos -> predict_dict[seed][data][method][exist_val]
        self.df_error = self.input_error()

    def input_error(self):
        self.results_input()
        df = pd.DataFrame(columns = ['data', 'seed', 'miss_val', 'method', 'MAE', 'RMSE'])

        for seed in self.seeds:
            for data in self.data_base:
                for exist_val in self.exist_val_percent:
                    for method in self.methods_base:
                        if (method == 'gnn_mdi'):
                            mae_grape = np.mean(np.abs(
                                self.dict_metadata[seed][data][method][exist_val]['outputs']['final_pred_test'] -
                                self.dict_metadata[seed][data][method][exist_val]['outputs']['label_test']))
                            rmse_grape = min(self.dict_metadata[seed][data][method][exist_val]['curves']['test_rmse'])
                            df.loc[len(df.index)] = [data, seed, exist_val, 'GRAPE', mae_grape, rmse_grape]
                        else:
                            method_name = method.capitalize()
                            if (method == 'knn' or method == 'svd'):
                                method_name = method_name.upper()
                            df.loc[len(df.index)] = [data, seed, exist_val, method_name,
                                                     self.dict_metadata[seed][data][method][exist_val]['mae'],
                                                     self.dict_metadata[seed][data][method][exist_val]['rmse']]

        return df

    def input_data(self, orig_data, input_data):
        flat_data = orig_data.flatten()
        estimatives = input_data['outputs']['final_pred_test'].flatten()

        flat_data[np.isnan(flat_data)] = estimatives
        return flat_data.reshape(orig_data.shape)

    def file_result(self, path):
        with open(path, 'rb') as file:
            results = pickle.load(file)
        return results

    def results_input(self):
        for seed in self.seeds:
            self.missing_dict[seed] = {}
            self.dict_metadata[seed] = {}
            self.filled_dict[seed] = {}
            self.predict_dict[seed] = {}

            # Resultados input  Baseline + GRAPE
            for data in self.data_base:
                self.dict_metadata[seed][data] = {}
                self.filled_dict[seed][data] = {}
                self.predict_dict[seed][data] = {}

                self.data_dict[data] = {}
                self.missing_dict[seed][data] = {}

                for method in self.methods_base:
                    self.dict_metadata[seed][data][method] = {}
                    self.filled_dict[seed][data][method] = {}
                    self.predict_dict[seed][data][method] = {}

                    path_label = f'GRAPE/uci/raw_data/{data}/data/data.csv'
                    df = pd.read_csv(path_label)
                    y = df.iloc[:, -1].values.reshape(-1, 1)

                    # infelizmente, pela implementação do GRAPE, ao inves de usar o scaler somente nos dados treino usa-se no conjunto inteiro
                    self.data_dict[str(data) + 'scaler'] = MinMaxScaler().fit(y)
                    scaler = MinMaxScaler()
                    self.data_dict[data] = pd.DataFrame(scaler.fit_transform(df), columns = df.columns)
                    if data == 'energy':
                        del self.data_dict[data]['Y1']

                    # ficou confuso mas não vai dar tempo de corrigir (miss_val X exist_val)
                    for miss_val in self.exist_val_percent:
                        path_baseline = f'GRAPE/uci/mdi_results/results/{method}_s{seed}/{data}/{miss_val}/result.pkl'
                        self.dict_metadata[seed][data][method][miss_val] = self.file_result(path_baseline)

                        path_grape = f'GRAPE/uci/y_results/gnn_s{seed}/{data}/{miss_val}/result.pkl'
                        self.predict_dict[seed][data][method][miss_val] = self.file_result(path_grape)

                        if method != 'gnn_mdi':
                            self.missing_dict[seed][data][miss_val] = self.dict_metadata[seed][data][method][miss_val][
                                'X_incomplete']
                            self.filled_dict[seed][data][method][miss_val] = \
                            self.dict_metadata[seed][data][method][miss_val]['X_filled']
                        else:
                            self.filled_dict[seed][data][method][miss_val] = self.input_data(
                                self.missing_dict[seed][data][miss_val],
                                self.dict_metadata[seed][data][method][miss_val])

                            # Funcao para salvar df

    # np.savetxt('concrete_75.txt', missing_dict[data_base][exist_val_percent], delimiter='\t', fmt='%.2f')

    # Funcao para checar similaridade dos dados VVVVVVVVVVV
    # np.sum(teste_base['X_incomplete'].flatten() == teste_knn['X_incomplete'].flatten()) / len(teste_knn['X_incomplete'].flatten())
    def plot_error_bars(self, metric = 'RMSE', exist_val_percent = 0.30):

        plt.figure(figsize = (15, 6))
        sns.barplot(data = self.df_error[self.df_error['miss_val'] == exist_val_percent], x = 'data', y = metric,
                    hue = 'method')
        plt.title(metric + f' - {100 - int(exist_val_percent * 100)}% missing values', fontsize = 14)
        plt.xlabel('Data source', fontsize = 12)
        plt.ylabel(metric)
        plt.xticks(fontsize = 10)

        x_min = 0.03
        x_max = 1 / len(self.data_base) - x_min
        for data in self.data_base:
            df_metric = self.df_error[
                (self.df_error['miss_val'] == exist_val_percent) & (self.df_error['data'] == data)]
            min_rmse = df_metric.groupby(by = 'method')[metric].mean().sort_values().iloc[0]
            scnd_rmse = df_metric.groupby(by = 'method')[metric].mean().sort_values().iloc[1]
            grape_rmse = df_metric[df_metric['method'] == 'GRAPE'].groupby(by = 'method')[metric].mean().iloc[0]
            plt.axhline(xmin = x_min - 0.03, xmax = x_max + 0.03, y = min_rmse, lw = 4, color = 'white')
            plt.axhline(xmin = x_min, xmax = x_max, y = scnd_rmse, linestyle = '--', lw = 2, color = 'red')
            plt.axhline(xmin = x_min, xmax = x_max, y = grape_rmse, linestyle = '--', lw = 2, color = 'blue')
            x_min += 1 / len(self.data_base)
            x_max += 1 / len(self.data_base)

        plt.axhline(xmin = 0, xmax = 0, y = 0, lw = 4, color = 'white', label = metric + ' Mínimo')
        plt.axhline(xmin = 0, xmax = 0, y = 0, linestyle = '--', lw = 2, color = 'red', label = metric + ' Baseline')
        plt.axhline(xmin = 0, xmax = 0, y = 0, linestyle = '--', lw = 2, color = 'blue', label = metric + ' GRAPE')
        plt.legend(title = 'Data')

        plt.tight_layout()
        plt.show()

    def plot_error_line(self, data, metric = 'RMSE'):
        methods = ['GRAPE', 'KNN', 'Mice', 'Mean', 'SVD', 'Spectral']
        result = self.df_error[self.df_error['data'] == data].groupby(['method', 'miss_val'])[
            metric].mean().reset_index()

        fig, ax = plt.subplots(1, 1, figsize = (15, 9))
        ax = fig.add_subplot(111, projection = '3d')

        for i, method in enumerate(methods):
            df_tmp = result.loc[result['method'] == method]
            x = 4 * [i + 1]
            y = 1 - df_tmp['miss_val']
            z = df_tmp[metric]
            ax.plot(x, y, z, label = f'Error {method}', marker = 'o')

        tick_labels = {(1 + index): algorit for index, algorit in enumerate(methods)}  # Define your own labels here

        ax.set_xticks(list(tick_labels.keys()))
        ax.set_xticklabels(list(tick_labels.values()))

        ax.set_xlabel('Algorithm')
        ax.set_ylabel('% Missing values')
        ax.set_zlabel(metric)
        ax.legend()
        ax.invert_xaxis()

        ax.set_title(metric + f" - {data.capitalize()}")

        plt.show()