"""
Snakemake workflow for gene upstream gene prediction.

Write by Hu Zhao, 2022.06.23
zhaohu44@gmail.com

Using xgboost, sklearn.
"""
import sys
import yaml
import pandas as pd
from os.path import join
import numpy as np
import joblib
import os
import ml as ml


def message(mes):
    sys.stderr.write("|---"+ mes + "\n")

tf_hdf = 'RAP_MSU_TF_df.h5'


express_f = 'gene_expression_df_renamed_mirna.csv' # expression matrix
anno_f = 'RAP_MSU_MIRNA_genes_sorted_bed_anno.h5' # gene function annotation file

njobs = 5

expression_df = pd.read_csv(express_f, index_col=0)
predict_list = ['Os06g0474200']

pculs_f = 'rice1479_cultivars_information.csv' # used cultivar information for prediction
pculs_info = pd.read_csv(pculs_f)
culs1 = set(pculs_info.loc[:,'Cultivar ID'])
culs_used = culs1

base_dir = 'Gene_TF_optuna'
message("the current dir is " +base_dir)
workdir:base_dir


rule all:
    input:
        expand('{gene}_LGBMRegressor.db', gene=predict_list),
        expand('{gene}_LGBMRegressor_res.h5', gene=predict_list),
        'all_res_summary.h5',

rule lightgbm:
    input:
        express_f = express_f,
    params:
        predict = '{gene}',
        train_repeat_times = 5,
        num_split = 5,
        njobs = njobs,
        up = 100,
        down = 100,
        score = 'R2',
        pred_repeat_times = 10,
        path = base_dir,
        num = '8268' #seed for prediction
    output:
        model = '{gene}_LGBMRegressor.db',
        model_res = '{gene}_LGBMRegressor_res.h5'
    run:
        res = ml.gene_TF(params.predict, culs_used, params.path, save_name = '{}'.format(params.predict), express_f=input.express_f, filter_number = None)
        res.make_dataset()
        res.data_split(seed = int(params.num))
        res.lgb_train(n_trials=50000, seed = int(params.num))
        res.perd_scores(name='LGBMRegressor')

rule summary_res:
    input:
        lgb = expand('{gene}_LGBMRegressor_res.h5', gene=predict_list),
    output:
        res = 'all_res_summary.h5'
    run:
        lgb_list = []
        anno_df = pd.read_hdf(anno_f)
        anno_df = anno_df.drop_duplicates(subset = ['gene'], keep='first')
        index_list = []
        for g in predict_list:
            name = g
            index_list.append(name)
            ld = pd.read_hdf('{}_LGBMRegressor_res.h5'.format(name), key='score')
            lgb_list.append(ld)
            feature_df = pd.DataFrame()
            for k in ['org_feature', 'shap_feature', 'shap_feature_norm']:
                if k == 'org_feature':
                    feature = pd.read_hdf('{}_LGBMRegressor_res.h5'.format(name), key=k)
                    feature.index = feature.Feature
                    feature_df.loc[:,'LGBM'] = feature.loc[:,'Value']
                elif k == 'shap_feature':
                    feature = pd.read_hdf('{}_LGBMRegressor_res.h5'.format(name), key=k)
                    feature_df.loc[:,'LGBM_SHAP'] = feature
                else:
                    feature = pd.read_hdf('{}_LGBMRegressor_res.h5'.format(name), key=k)
                    feature_df.loc[:,'LGBM_SHAP_norm'] = feature
            feature_index = list(feature_df.index)
            feature_df.loc[:,'symbol'] = anno_df.Osymbols
            feature_df.loc[:,'TF'] = anno_df.TF
            feature_df.loc[:,'description'] = anno_df.description
            feature_df = feature_df.sort_values(['LGBM_SHAP'], ascending=False)
            feature_df.to_hdf(output.res, key=name)
        lgb_df = pd.concat(lgb_list, axis=0)
        lgb_df.index = index_list
        lgb_df.to_hdf(output.res, key='LGBM_scores')




