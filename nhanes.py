import pdb
import glob
import copy
import os
import pickle
import joblib

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats
import sklearn.feature_selection


class FeatureColumn:
    def __init__(self, category, field, preprocessor, args=None, cost=None):
        self.category = category
        self.field = field
        self.preprocessor = preprocessor
        self.args = args
        self.data = None
        self.cost = cost


class NHANES:
    def __init__(self, db_path=None, columns=None):
        self.db_path = db_path
        self.columns = columns  # Depreciated
        self.dataset = None  # Depreciated
        self.column_data = None
        self.column_info = None
        self.df_features = None
        self.df_targets = None
        self.costs = None

    def process(self):
        '''
        Process the data
        '''
        df = None
        cache = {}
        # collect relevant data
        df = []
        for fe_col in self.columns:
            sheet = fe_col.category
            field = fe_col.field
            data_files = glob.glob(self.db_path+sheet+'/*.XPT')
            df_col = []
            for dfile in data_files:
                print(80*' ', end='\r')
                print('\rProcessing: ' + dfile.split('/')[-1], end='')
                # read the file
                if dfile in cache:
                    df_tmp = cache[dfile]
                else:
                    df_tmp = pd.read_sas(dfile)
                    cache[dfile] = df_tmp
                # skip of there is no SEQN
                if 'SEQN' not in df_tmp.columns:
                    continue
                # df_tmp.set_index('SEQN')
                # skip if there is nothing interseting there
                sel_cols = set(df_tmp.columns).intersection([field])
                if not sel_cols:
                    continue
                else:
                    df_tmp = df_tmp[['SEQN'] + list(sel_cols)]
                    df_tmp.set_index('SEQN', inplace=True)
                    df_col.append(df_tmp)

            try:
                df_col = pd.concat(df_col)
            except:
                #raise Error('Failed to process' + field)
                raise Exception('Failed to process ' + field)
            df.append(df_col)
        df = pd.concat(df, axis=1)
        #df = pd.merge(df, df_sel, how='outer')
        # do preprocessing steps
        df_proc = []  # [df['SEQN']]
        for fe_col in self.columns:
            field = fe_col.field
            fe_col.data = df[field].copy()
            # do preprocessing
            if fe_col.preprocessor is not None:
                prepr_col = fe_col.preprocessor(df[field], fe_col.args)
            else:
                prepr_col = df[field]
            # handle the 1 to many
            if (len(prepr_col.shape) > 1):
                fe_col.cost = [fe_col.cost] * prepr_col.shape[1]
            else:
                fe_col.cost = [fe_col.cost]
            df_proc.append(prepr_col)
        self.dataset = pd.concat(df_proc, axis=1)
        return self.dataset


# Preprocessing functions
def preproc_onehot(df_col, args=None):
    return pd.get_dummies(df_col, prefix=df_col.name, prefix_sep='#')


def preproc_real(df_col, args=None):
    if args is None:
        args = {'cutoff': np.inf}
    # other answers as nan
    df_col[df_col > args['cutoff']] = np.nan
    # nan replaced by mean
    df_col[pd.isna(df_col)] = df_col.mean()
    # statistical normalization
    df_col = (df_col-df_col.mean()) / df_col.std()
    return df_col


def preproc_impute(df_col, args=None):
    # nan replaced by mean
    df_col[pd.isna(df_col)] = df_col.mean()
    return df_col


def preproc_cut(df_col, bins):
    # limit values to the bins range
    df_col = df_col[df_col >= bins[0]]
    df_col = df_col[df_col <= bins[-1]]
    return pd.cut(df_col.iloc[:, 0], bins, labels=False)


def preproc_dropna(df_col, args=None):
    df_col.dropna(axis=0, how='any', inplace=True)
    return df_col

#### Add your own preprocessing functions ####


def PCA(X):
    """
    Perform Principal Component Analysis.
    This version uses SVD for better numerical performance when d >> n.

    Parameters
    --------------------
        X      -- numpy array of shape (n,d), features

    Returns
    --------------------
        U      -- numpy array of shape (d,d), d d-dimensional eigenvectors
                  each column is a unit eigenvector; columns are sorted by eigenvalue
        mu     -- numpy array of shape (d,), mean of input data X
    """
    n, d = X.shape
    mu = np.mean(X, axis=0)
    x, l, v = np.linalg.svd(X-mu)
    l = np.hstack([l, np.zeros(v.shape[0] - l.shape[0], dtype=float)])
    U = np.array([vi/1.0
                  for (li, vi)
                  in sorted(zip(l, v), reverse=True, key=lambda x: x[0])]).T
    return U, mu

# Dataset loader


class Dataset():
    """ 
    Dataset manager class
    """

    def __init__(self, data_path=None):
        """
        Class intitializer.
        """
        # set database path
        if data_path == None:
            self.data_path = './run_data/'
        else:
            self.data_path = data_path
        # feature and target vecotrs
        self.features = None
        self.targets = None
        self.costs = None

    #### Custom dataset loaders ####

    def load_cancer(self, opts=None):
        columns = [
            # TARGET: Ever told you had cancer or malignancy
            FeatureColumn('Questionnaire', 'MCQ220',
                          None, None),
            # Gender
            FeatureColumn('Demographics', 'RIAGENDR',
                          preproc_real, None),
            # Age at time of screening
            FeatureColumn('Demographics', 'RIDAGEYR',
                          preproc_real, None),
            # Recode of reported race and Hispanic origin
            # information, with Non-Hispanic Asian Category
            FeatureColumn('Demographics', 'RIDRETH3',
                          preproc_onehot, None),
            # Race/ethnicity
            FeatureColumn('Demographics', 'RIDRETH1',
                          preproc_onehot, None),
            # Annual household income
            FeatureColumn('Demographics', 'INDHHIN2',
                          preproc_real, {'cutoff': 11}),
            # Annual family income
            FeatureColumn('Demographics', 'INDFMIN2',
                          preproc_real, {'cutoff': 11}),
            # Ratio of family income to poverty
            FeatureColumn('Demographics', 'INDFMPIR',
                          preproc_real, {'cutoff': 5}),
            # Education level
            FeatureColumn('Demographics', 'DMDEDUC2',
                          preproc_real, {'cutoff': 5}),
            # Alcohol Consumed Day 1 (g)
            FeatureColumn('Dietary', 'DR1TALCO',
                          preproc_real, None),
            # Kcal Consumed Day 1 (g)
            FeatureColumn('Dietary', 'DR1TKCAL',
                          preproc_real, None),
            # Sugar Consumed Day 1 (g)
            FeatureColumn('Dietary', 'DR1TSUGR',
                          preproc_real, None),
            # Alcohol Consumed Day 2 (g)
            FeatureColumn('Dietary', 'DR2TALCO',
                          preproc_real, None),
            # Kcal Consumed Day 2 (g)
            FeatureColumn('Dietary', 'DR2TKCAL',
                          preproc_real, None),
            # Sugar Consumed Day 3 (g)
            FeatureColumn('Dietary', 'DR3TSUGR',
                          preproc_real, None),
            # On a weight loss / low calorie diet?
            FeatureColumn('Dietary', 'DRQSDT1',
                          preproc_real, None),
            # BMI
            FeatureColumn('Examination', 'BMXBMI',
                          preproc_real, None),
            # Waist
            FeatureColumn('Examination', 'BMXWAIST',
                          preproc_real, None),
            # Height
            FeatureColumn('Examination', 'BMXHT',
                          preproc_real, None),
            # Upper Leg Length
            FeatureColumn('Examination', 'BMXLEG',
                          preproc_real, None),
            # Weight
            FeatureColumn('Examination', 'BMXWT',
                          preproc_real, None),
            # Total Cholesterol
            FeatureColumn('Laboratory', 'LBXTC',
                          preproc_real, None),
            # Alcohol consumption
            FeatureColumn('Questionnaire', 'ALQ101',
                          preproc_real, {'cutoff': 2}),
            FeatureColumn('Questionnaire', 'ALQ120Q',
                          preproc_real, {'cutoff': 365}),
            # Vigorous work activity
            FeatureColumn('Questionnaire', 'PAQ605',
                          preproc_real, {'cutoff': 2}),
            FeatureColumn('Questionnaire', 'PAQ620',
                          preproc_real, {'cutoff': 2}),
            FeatureColumn('Questionnaire', 'PAQ180',
                          preproc_real, {'cutoff': 4}),
            FeatureColumn('Questionnaire', 'PAD615',
                          preproc_real, {'cutoff': 780}),
            # Doctor told overweight (risk factor)
            FeatureColumn('Questionnaire', 'MCQ160J',
                          preproc_onehot, {'cutoff': 2}),
            # Sleep
            FeatureColumn('Questionnaire', 'SLD010H',
                          preproc_real, {'cutoff': 12}),
            # Smoking
            FeatureColumn('Questionnaire', 'SMQ020',
                          preproc_onehot, None),
            FeatureColumn('Questionnaire', 'SMD030',
                          preproc_real, {'cutoff': 72}),
            # Blood relatives with arthritis
            FeatureColumn('Questionnaire', 'MCQ250D',
                          preproc_onehot, {'cutoff': 2}),
            # joint pain/aching/stiffness in past year
            FeatureColumn('Questionnaire', 'MPQ010',
                          preproc_onehot, {'cutoff': 2}),
            # symptoms began only because of injury
            FeatureColumn('Questionnaire', 'MPQ030',
                          preproc_onehot, {'cutoff': 2}),
            # how long experiencing pain
            FeatureColumn('Questionnaire', 'MPQ110',
                          preproc_real, {'cutoff': 4}),
        ]
        nhanes_dataset = NHANES(self.data_path, columns)
        df = nhanes_dataset.process()
        fe_cols = df.drop(['MCQ220'], axis=1)
        features = fe_cols.values
        target = df['MCQ220'].values
        # remove nan labeled samples
        inds_valid = ~ np.isnan(target)
        features = features[inds_valid]
        target = target[inds_valid]

        # Put each person in the corresponding bin
        targets = np.zeros(target.shape[0])
        targets[target == 1] = 0  # yes arthritis
        targets[target == 2] = 1  # no arthritis

       # random permutation
        perm = np.random.permutation(targets.shape[0])
        self.features = features[perm]
        self.targets = targets[perm]
        self.costs = [c.cost for c in columns[1:]]
        self.costs = np.array(
            [item for sublist in self.costs for item in sublist])
