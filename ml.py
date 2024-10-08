def feature_importances_shap_values(shap_values, features):
    '''
    Prints the feature importances based on SHAP values in an ordered way
    shap_values -> The SHAP values calculated from a shap.Explainer object
    features -> The name of the features, on the order presented to the explainer
    code from https://github.com/vinyluis/Articles/tree/main/Boruta%20SHAP
    '''
    # Calculates the feature importance (mean absolute shap value) for each feature
    importances = []
    for i in range(shap_values.values.shape[1]):
        importances.append(np.mean(np.abs(shap_values.values[:, i])))
    # Calculates the normalized version
    importances_norm = softmax(importances)
    # Organize the importances and columns in a dictionary
    feature_importances = {fea: imp for imp, fea in zip(importances, features)}
    feature_importances_norm = {fea: imp for imp, fea in zip(importances_norm, features)}
    # Sorts the dictionary
    feature_importances = {k: v for k, v in sorted(feature_importances.items(), key=lambda item: item[1], reverse = True)}
    feature_importances_norm= {k: v for k, v in sorted(feature_importances_norm.items(), key=lambda item: item[1], reverse = True)}
    return [feature_importances, feature_importances_norm]

class ml_data():
    def __init__(self, Xdata, Ydata, data_save_path, save_name):
        self.Xdata = Xdata
        self.Ydata = Ydata
        self.data_save_path = data_save_path
        self.save_name = save_name

    def data_split(self, test_size=0.25, seed=0):
        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(self.Xdata, self.Ydata, test_size=test_size, random_state= seed)
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test


    def xgb_train(self, n_cv=5, n_repeats=5, n_trials=500, seed = 123):
        import joblib
        import optuna
        from optuna import trial
        import xgboost as xgb
        import os
        from sklearn.metrics import make_scorer, r2_score, mean_squared_error
        os.chdir(self.data_save_path)
        pruner = optuna.pruners.MedianPruner(n_warmup_steps=10)
        study = optuna.create_study(study_name='XGBRegressor', pruner=pruner, direction='minimize',storage="sqlite:///{}_XGBRegressor.db".format(self.save_name), load_if_exists=True)
        study.optimize(lambda trial: xg_objective(trial, self.X_train, self.y_train, n_cv, n_repeats, seed), n_trials= n_trials, gc_after_trial=True)
        print("Number of finished trials: {}".format(len(study.trials)))

    def lgb_train(self, n_cv=5, n_repeats=5, n_trials=500, seed = 123):
        import optuna
        import lightgbm as lgb
        import os
        os.chdir(self.data_save_path)
        pruner = optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=0, interval_steps=1)
        study = optuna.create_study(study_name='LGBMRegressor', pruner=pruner, direction='minimize',storage="sqlite:///{}_LGBMRegressor.db".format(self.save_name), load_if_exists=True)
        study.optimize(lambda trial: lg_objective(trial, self.X_train, self.y_train, n_cv, n_repeats, seed), n_trials= n_trials, gc_after_trial=True)
        print("Number of finished trials: {}".format(len(study.trials)))
    def perd_scores(self, name='XGBRegressor'):
        """
        name should be XGBRegressor or LGBRegressor
        """
        import optuna
        import os
        import shap
        import xgboost as xgb
        import lightgbm as lgb
        os.chdir(self.data_save_path)
        if name == 'XGBRegressor':  ##XGBoost use num_boost_round in train, but use n_estimators in fit, (:
            study = optuna.load_study(study_name= name, storage="sqlite:///{}_XGBRegressor.db".format(self.save_name))
            params = study.best_params
            this_study = xgb.XGBRegressor(**params)
            this_study.fit(self.X_train, self.y_train)
            feature_imp = pd.DataFrame(sorted(zip(this_study.feature_importances_,self.X_train.columns),reverse=True), columns=['Value','Feature'])
            models = {f'XGBRegressor {params}': this_study,}
            df = evaluate_model(models, self.X_train, self.X_test, self.y_train, self.y_test)
            df.to_hdf('{}_XGBRegressor_res.h5'.format(self.save_name),key='score')
            feature_imp.to_hdf('{}_XGBRegressor_res.h5'.format(self.save_name),key='org_feature')

            explainer = shap.Explainer(this_study.predict, self.Xdata)
            shap_values = explainer(self.Xdata, max_evals=1000)
            shap_importance, shap_importance_norm = feature_importances_shap_values(shap_values, self.X_train.columns)
            pd.Series(shap_importance).to_hdf('{}_XGBRegressor_res.h5'.format(self.save_name),key='shap_feature')
            pd.Series(shap_importance_norm).to_hdf('{}_XGBRegressor_res.h5'.format(self.save_name),key='shap_feature_norm')
            np.savez('{}_XGBRegressor_shap_values.npz'.format(self.save_name), shap_values)

        else:
            study = optuna.load_study(study_name= name, storage="sqlite:///{}_LGBMRegressor.db".format(self.save_name))
            params = study.best_params
            params.update({"verbosity": -1,})
            this_study = lgb.LGBMRegressor(**params)
            this_study.fit(self.X_train, self.y_train)
            feature_imp = pd.DataFrame(sorted(zip(this_study.feature_importances_,self.X_train.columns), reverse=True), columns=['Value','Feature'])
            models = {f'LGBMRegressor {params}': this_study,}
            df = evaluate_model(models, self.X_train, self.X_test, self.y_train, self.y_test)
            df.to_hdf('{}_LGBMRegressor_res.h5'.format(self.save_name),key='score')
            feature_imp.to_hdf('{}_LGBMRegressor_res.h5'.format(self.save_name),key='org_feature')

            explainer = shap.Explainer(this_study.predict, self.Xdata)
            shap_values = explainer(self.Xdata, max_evals=50000)
            shap_importance, shap_importance_norm = feature_importances_shap_values(shap_values, self.X_train.columns)
            pd.Series(shap_importance).to_hdf('{}_LGBMRegressor_res.h5'.format(self.save_name),key='shap_feature')
            pd.Series(shap_importance_norm).to_hdf('{}_LGBMRegressor_res.h5'.format(self.save_name),key='shap_feature_norm')
            np.savez('{}_LGBMRegressor_shap_values.npz'.format(self.save_name), shap_values)

class gene_TF(ml_data):
    def __init__(self, gene, accessions, data_save_path, pred_f=None, save_name = None, express_f= None, anno_f = None, filter_number = None):
        self.gene = gene
        self.data_save_path = data_save_path
        self.express_f = express_f
        self.accessions = accessions
        self.anno_f = anno_f
        anno_df = pd.read_hdf(anno_f)
        anno_df = anno_df.drop_duplicates(subset = ['gene'], keep='first')
        self.anno_df = anno_df
        express_df = pd.read_csv(express_f, index_col=0)
        self.express_df = express_df
        if pred_f is None:
            self.pred_df = express_df
        else:
            self.pred_df = pd.read_csv(pred_f, index_col=0)
        if save_name is None:
            self.save_name = gene
        else:
            self.save_name = save_name
        self.filter_number = filter_number
        if os.path.isdir(data_save_path):
            pass
        else:
            os.makedirs(data_save_path)

    def make_dataset(self):
        TF_df = self.anno_df.dropna(subset=['TF'])
        TF_genes = list((set(TF_df.gene) & set(self.express_df.index)) - set([self.gene]) )
        culs = list(set(self.accessions) & set(self.express_df.columns) & set(self.pred_df.columns))
        print('{} cultivars used'.format(len(culs)))
        X_df = self.express_df.loc[TF_genes, culs].T
        Y = self.pred_df.loc[self.gene, culs]
        if self.filter_number is not None:
            used_TFs = list(X_df.corrwith(Y).abs().sort_values(ascending= False).iloc[:self.filter_number].index)
            self.Xdata = X_df.loc[:, used_TFs]
        else:
            self.Xdata = X_df
        self.Ydata = Y
