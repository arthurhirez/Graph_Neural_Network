import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold

from sklearn.linear_model import LinearRegression, ElasticNetCV, LassoCV, Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor

from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.metrics import mean_squared_error, explained_variance_score, max_error, mean_absolute_error


class Experiment_Predict():
    def __init__(self, data_raw, input_grape, input_baseline,
                 seed_list = None, data_list = None, input_list = None,
                 value_list = None):

        self.seeds = [0] if seed_list is None else seed_list
        self.data_base = ['concrete'] if data_list is None else data_list
        self.method_input = ['gnn_mdi', 'knn'] if input_list is None else input_list
        self.exist_val_percent = [0.30, 0.50, 0.70, 0.90] if value_list is None else value_list

        self.target_var = {'concrete': 'compr_strength', 'energy': 'Y2', 'power': 'PE'}

        self.algotithm_baseline = [LinearRegression(fit_intercept = False),
                                   ElasticNetCV(cv = 10, random_state = 12),
                                   PolynomialFeatures(),
                                   Lasso(),
                                   DecisionTreeRegressor(),
                                   RandomForestRegressor(random_state = 12),
                                   GradientBoostingRegressor(random_state = 12),
                                   KNeighborsRegressor(n_neighbors = 4, weights = 'uniform'),
                                   KNeighborsRegressor(n_neighbors = 4, weights = 'distance')
                                   ]
        self.data_dict = data_raw
        self.predict_dict = input_grape
        self.filled_dict = input_baseline
        self.dict_split = {}
        self.method_dict = {}
        self.df_result = pd.DataFrame()


    def data_split(self, df, target):
        X = df.drop(columns = target)
        y = df[target]
        return X, y

    def split_inputed(self, seed, method, data, exist_val, target, X_tr, y_tr, X_ts, y_ts):

        columns_name = self.data_dict[data].iloc[:, :-1].columns
        method_data = pd.DataFrame(self.filled_dict[seed][data][method][exist_val], columns = columns_name)
        method_data[target] = self.data_dict[data].iloc[:, -1]

        # Treino e teste do conjunto de Treino
        treino_split_meth = train_test_split(method_data, method_data[target], test_size = 0.20, random_state = 12)
        X_tr_meth, y_tr_meth = self.data_split(treino_split_meth[0], target)
        X_ts_meth, y_ts_meth = self.data_split(treino_split_meth[1], target)

        def check_indexes(original, method):
            or_idx = set(original.index)
            mt_idx = set(method.index)
            if or_idx != mt_idx:
                print("ERROR SPLITING DATA!!")
                return 1
            return 0

        flag = 0
        flag += check_indexes(X_tr_meth, X_tr)
        flag += check_indexes(X_ts_meth, X_ts)
        flag += check_indexes(y_tr_meth, y_tr)
        flag = check_indexes(y_ts_meth, y_ts)

        if not flag:
            self.dict_split[seed][data][method][exist_val]['X_tr'] = X_tr_meth
            self.dict_split[seed][data][method][exist_val]['X_ts'] = X_ts_meth
            self.dict_split[seed][data][method][exist_val]['y_tr'] = y_tr_meth
            self.dict_split[seed][data][method][exist_val]['y_ts'] = y_ts_meth
        else:
            print("ERROR: CHECK SPLITTED DATA!")
            self.dict_split[seed][data][method][exist_val] = None

    def run_baseline(self, seed, data_id, exist_val, algorithms, x_treino, x_teste, y_treino, y_teste, data,
                     plot_lasso = False):
        algotithm_label = ['Linear Regression',
                           'Elastic Net CV',
                           'Polynomial Features',
                           'Lasso',
                           'Decision Tree',
                           'Random Forest',
                           'Gradient Boosting',
                           'KNeighbors Uniform',
                           'KNeighbors Distance'
                           ]

        df_result = pd.DataFrame(
            columns = ['seed', 'data', 'input_alg', 'exist_val', 'regression_alg', 'MSE', 'MAE', 'Max_error',
                       'Explained variance'])

        for i, alg in enumerate(algorithms):
            model_poly = Pipeline([('poly', PolynomialFeatures(degree = 3, interaction_only = True)),
                                   ('linear', LinearRegression(fit_intercept = False))])

            if str(alg) == 'GRAPE_y':
                y_tes = self.data_dict[data + 'scaler'].transform(y_teste.reshape(-1, 1))
                y_trei = self.data_dict[data + 'scaler'].transform(y_treino.reshape(-1, 1))
                df_result.loc[str(alg)] = [seed, data, data_id, exist_val, 'Grape_y',
                                           mean_squared_error(y_true = y_tes, y_pred = y_trei),
                                           mean_absolute_error(y_true = y_tes, y_pred = y_trei),
                                           max_error(y_true = y_tes, y_pred = y_trei),
                                           explained_variance_score(y_true = y_tes, y_pred = y_trei),
                                           ]
                # model_dict[str(alg)] = {'estimador' : 'grape_gnn', 'y_pred' : y_treino}
                return df_result

            elif str(alg) == "PolynomialFeatures()":
                estimador = model_poly.fit(x_treino, y_treino)
                y_pred = estimador.predict(x_teste)

            elif str(alg) == "Lasso()":

                X_tr_poly = model_poly.fit(x_treino, y_treino).named_steps['poly'].transform(x_treino)
                model_lassocv = make_pipeline(LassoCV(cv = 10, max_iter = 10000, tol = 1e-3)).fit(X_tr_poly,
                                                                                                  y_treino)

                lasso = model_lassocv[-1]
                alpha_CV = lasso.alpha_

                if plot_lasso:
                    plt.subplots(1, 1, figsize = (10, 6))
                    plt.semilogx(lasso.alphas_, lasso.mse_path_, linestyle = ":")
                    plt.plot(
                        lasso.alphas_,
                        lasso.mse_path_.mean(axis = -1),
                        color = "black",
                        label = "Average across the folds",
                        linewidth = 2)
                    plt.axvline(lasso.alpha_, linestyle = "--", color = "black",
                                label = f"alpha: CV estimate = {lasso.alpha_:.2e}")

                    plt.xlabel(r"$\alpha$")
                    plt.ylabel("Mean square error")
                    plt.legend()
                    plt.title(f"Mean square error on each fold  - LassoCV")
                    plt.show()

                model_lasso = Pipeline(steps = [('Lasso', Lasso(alpha = 1e-4, max_iter = 10000, tol = 1e-3))])

                x_teste_lasso = model_poly.named_steps['poly'].transform(x_teste)
                estimador = model_lasso.fit(X_tr_poly, y_treino)
                y_pred = estimador.predict(x_teste_lasso)
            else:
                estimador = alg.fit(x_treino, y_treino)
                y_pred = estimador.predict(x_teste)

            df_result.loc[str(alg)] = [seed, data, data_id, exist_val, algotithm_label[i],
                                       mean_squared_error(y_true = y_teste, y_pred = y_pred),
                                       mean_absolute_error(y_true = y_teste, y_pred = y_pred),
                                       max_error(y_true = y_teste, y_pred = y_pred),
                                       explained_variance_score(y_true = y_teste, y_pred = y_pred),
                                       ]
        return df_result

    def compile_predict(self):
        for seed in self.seeds:
            self.dict_split[seed] = {}
            self.method_dict[seed] = {}
            for data in self.data_base:
                self.method_dict[seed][data] = {}
                self.dict_split[seed][data] = {}

                original_data = self.data_dict[data]
                if (data == 'concrete'):
                    original_data = original_data.rename(
                        columns = {'concrete_compressive_strength': self.target_var[data]})

                # Treino e teste do conjunto de Treino
                treino_split = train_test_split(original_data, original_data[self.target_var[data]],
                                                test_size = 0.20, random_state = 12)
                X_tr, y_tr = self.data_split(treino_split[0], self.target_var[data])
                X_ts, y_ts = self.data_split(treino_split[1], self.target_var[data])

                plt_lasso = False
                if seed == 0: plt_lasso = True
                df_baseline = self.run_baseline(seed, "original", 1, self.algotithm_baseline, X_tr, X_ts, y_tr,
                                                y_ts, data = data, plot_lasso = plt_lasso)
                self.df_result = pd.concat([self.df_result, df_baseline], ignore_index = True)

                len_data = len(X_tr) + len(X_ts)
                if seed == self.seeds[-1]:
                    print(
                        f"Conjunto de Dados: '{data.capitalize()}':\nTreino:\t{len(X_tr)}\t({100 * len(X_tr) / len_data :.0f}%)\nTeste:\t{len(X_ts)}\t({100 * len(X_ts) / len_data :.0f}%)\nTotal:\t{len_data}\n\n")

                for method in self.method_input:
                    self.method_dict[seed][data][method] = {}
                    self.dict_split[seed][data][method] = {}

                    for exist_val in self.exist_val_percent:
                        self.method_dict[seed][data][method][exist_val] = {}
                        self.dict_split[seed][data][method][exist_val] = {}

                        self.split_inputed(seed, method, data, exist_val, self.target_var[data], X_tr, y_tr, X_ts,
                                           y_ts)
                        method_split = self.dict_split[seed][data][method][exist_val]
                        df_method = self.run_baseline(seed, str(method), exist_val, self.algotithm_baseline,
                                                      method_split['X_tr'], method_split['X_ts'],
                                                      method_split['y_tr'], method_split['y_ts'], data)
                        self.method_dict[seed][data][method][exist_val]['df'] = df_method

                        if method == 'gnn_mdi':
                            df_gnn = self.predict_dict[seed][data]['gnn_mdi'][exist_val]['outputs']
                            df_gnn_y = self.run_baseline(seed, "GNN_input", exist_val, ["GRAPE_y"],
                                                         df_gnn['pred_test'], df_gnn['label_test'],
                                                         df_gnn['pred_test'], df_gnn['label_test'], data)
                            self.method_dict[seed][data][method][exist_val]['df'] = pd.concat([df_gnn_y, df_method],
                                                                                              ignore_index = True)

            for seed_dict in self.method_dict.values():
                for method_data in seed_dict.values():
                    for exist_val_data in method_data.values():
                        for df in exist_val_data.values():
                            self.df_result = pd.concat([self.df_result, df['df']], ignore_index = True)

            self.df_result.reset_index(drop = True, inplace = True)

    def plot_predict_error(self, data, metric):
        df = self.df_result[(self.df_result['data'] == data) & (self.df_result['input_alg'] != 'original')]
        c = 50

        fig, ax = plt.subplots(1, 1, figsize = (15, 9))
        ax = fig.add_subplot(111, projection = '3d')
        for i, input in enumerate(df['input_alg'].unique()):
            df_tmp = df.loc[(df['input_alg'] == input)].groupby(by = ['regression_alg', 'exist_val', 'input_alg'])[
                metric].mean().reset_index()
            for j, regression in enumerate(df['regression_alg'].unique()):
                if (input != "GNN_input" and regression != "Grape_y"):
                    x = 4 * [c * i + j]
                    y = 1 - df_tmp[df_tmp['regression_alg'] == regression]['exist_val']
                    z = df_tmp[df_tmp['regression_alg'] == regression][metric]
                    ax.plot(x, y, z, label = f'Error {input}', marker = 'o')
                elif (input == "GNN_input"):
                    x = 4 * [c * i]
                    y = 1 - df_tmp['exist_val']
                    z = df_tmp[metric]
                    ax.plot(x, y, z, label = f'Error {input}', marker = 'o')

        tick_labels = {(c * index): algorit for index, algorit in enumerate(df['input_alg'].unique())}

        ax.set_xticks(list(tick_labels.keys()))
        ax.set_xticklabels(list(tick_labels.values()))
        ax.set_xlabel('Algorithm')
        ax.set_ylabel('% Missing values')
        ax.set_zlabel(metric)
        # ax.legend()
        ax.invert_xaxis()
        ax.set_title(metric + f" - {data.capitalize()}")

        plt.show()

    def plot_predict_bars(self, data, exist_val, metric):
        df_pred = self.df_result[(self.df_result['data'] == data)]

        df_err = df_pred[(df_pred['exist_val'] == exist_val)]

        palette = sns.color_palette("Set2", n_colors = 12)
        figure, ax = plt.subplots(1, 1, figsize = (15, 6))

        sns.barplot(data = df_pred[df_pred['exist_val'] == 1], x = 'input_alg', y = metric, ax = ax,
                    color = palette[0])
        sns.barplot(data = df_err[df_err['regression_alg'] == 'Grape_y'], x = 'input_alg', y = metric, ax = ax,
                    color = palette[-1])

        sns.barplot(data = df_err[df_err['regression_alg'] != 'Grape_y'], x = 'input_alg', y = metric,
                    hue = 'regression_alg', ax = ax, palette = palette[2:-1])

        plt.title(metric + f' - {100 - int(exist_val * 100)}% missing values - "{data.capitalize()}"',
                  fontsize = 14)
        ax.set_xlabel('Input method', fontsize = 12)
        ax.set_ylabel(metric)

        x_min = 0.02
        x_max = 0.23

        df_metric = df_err.groupby(by = ['regression_alg', 'exist_val', 'input_alg'])[metric].mean().reset_index()

        min_rmse = df_metric[df_metric['regression_alg'] != 'Grape_y'][metric].sort_values().iloc[0]
        ax.axhline(xmin = 0, xmax = 1, y = min_rmse, linestyle = '--', lw = 2, color = 'red')

        grape_rmse = df_metric[df_metric['regression_alg'] == 'Grape_y'][metric].sort_values().iloc[0]
        ax.axhline(xmin = 0, xmax = 1, y = grape_rmse, linestyle = '--', lw = 4, color = 'white')

        ax.axhline(xmin = 0, xmax = 0, y = 0, lw = 4, color = 'white', label = metric + ' GRAPE')
        ax.axhline(xmin = 0, xmax = 0, y = 0, linestyle = '--', lw = 2, color = 'red', label = metric + ' Baseline')

        plt.legend(title = 'Data', bbox_to_anchor = (1, 1))

        plt.tight_layout()
        plt.show()