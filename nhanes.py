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
            df_col = df_col.loc[~df_col.index.duplicated()]
            df.append(df_col)
        df = pd.concat(df, axis=1)
        # df = pd.merge(df, df_sel, how='outer')
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


def preproc_cutoff(df_col, cutoff_val=None):
    if cutoff_val is None:
        return df_col
    # other answers as nan
    df_col[df_col > cutoff_val] = np.nan
    df_col = df_col.dropna(how='any')
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
            # Sugar Consumed Day 2 (g)
            FeatureColumn('Dietary', 'DR2TSUGR',
                          preproc_real, None),
            # On a weight loss / low calorie diet?
            FeatureColumn('Dietary', 'DRQSDT1',
                          preproc_onehot, None),
            # 60 second heart-rate
            FeatureColumn('Examination', 'BPXCHR',
                          preproc_real, None),
            # 60 second pulse
            FeatureColumn('Examination', 'BPXPLS',
                          preproc_real, None),
            # pulse regular or irregular
            FeatureColumn('Examination', 'BPXPULS',
                          preproc_onehot, None),
            ### Added features ###

            # Urinary arsenic, total (ug/L)
            FeatureColumn('Laboratory', 'URXUAS', preproc_real,
                          None),
            # Blood lead (ug/dL)
            FeatureColumn('Laboratory', 'LBXBPB', preproc_real,
                          None),
            # Blood cadmium (ug/L)
            FeatureColumn('Laboratory', 'LBXBCD', preproc_real,
                          None),
            # Blood mercury, total (ug/L)
            FeatureColumn('Laboratory', 'LBXTHG', preproc_real,
                          None),
            # Blood selenium (ug/L)
            FeatureColumn('Laboratory', 'LBXBSE', preproc_real,
                          None),
            # Blood manganese (ug/L)
            FeatureColumn('Laboratory', 'LBXBMN', preproc_real,
                          None),
            # Urine Mercury (ng/mL)
            FeatureColumn('Laboratory', 'URXUHG', preproc_real,
                          None),
            # Barium, urine (ug/L)
            FeatureColumn('Laboratory', 'URXUBA', preproc_real,
                          None),
            # Cadmium, urine (ug/L)
            FeatureColumn('Laboratory', 'URXUCD', preproc_real,
                          None),
            # Cobalt, urine (ug/L)
            FeatureColumn('Laboratory', 'URXUCO', preproc_real,
                          None),
            # Cesium, urine (ug/L)
            FeatureColumn('Laboratory', 'URXUCS', preproc_real,
                          None),
            # Molybdenum, urine (ug/L)
            FeatureColumn('Laboratory', 'URXUMO', preproc_real,
                          None),
            # Manganese, urine (ug/L)
            FeatureColumn('Laboratory', 'URXUMN', preproc_real,
                          None),
            # Lead, urine (ug/L)
            FeatureColumn('Laboratory', 'URXUPB', preproc_real,
                          None),
            # Antimony, urine (ug/L)
            FeatureColumn('Laboratory', 'URXUSB', preproc_real,
                          None),
            # Tin, urine (ug/L)
            FeatureColumn('Laboratory', 'URXUSN', preproc_real,
                          None),
            # Strontium, urine (ug/L)
            FeatureColumn('Laboratory', 'URXUSR', preproc_real,
                          None),
            # Thallium, urine (ug/L)
            FeatureColumn('Laboratory', 'URXUTL', preproc_real,
                          None),
            # Tungsten, urine (ug/L)
            FeatureColumn('Laboratory', 'URXUTU', preproc_real,
                          None),
            # Uranium, urine (ug/L)
            FeatureColumn('Laboratory', 'URXUUR', preproc_real,
                          None),
            # Doctor ever said you were overweight
            FeatureColumn('Questionnaire', 'MCQ080', preproc_onehot, None),
            # Doctor told you to lose weight
            FeatureColumn('Questionnaire', 'MCQ365A', preproc_onehot, None),
            # Doctor told you to exercise
            FeatureColumn('Questionnaire', 'MCQ365B', preproc_onehot, None),
            # Doctor told you to reduce fat/calories
            FeatureColumn('Questionnaire', 'MCQ365D', preproc_onehot, None),
            # Doctor told you to reduce salt in diet
            FeatureColumn('Questionnaire', 'MCQ365C', preproc_onehot, None),
            # Are you now controlling or losing weight
            FeatureColumn('Questionnaire', 'MCQ370A', preproc_onehot, None),
            # Are you now increasing exercise
            FeatureColumn('Questionnaire', 'MCQ370B', preproc_onehot, None),
            # Are you now reducing salt in diet
            FeatureColumn('Questionnaire', 'MCQ370C', preproc_onehot, None),
            # Are you now reducing fat in diet
            FeatureColumn('Questionnaire', 'MCQ370D', preproc_onehot, None),
            # Family monthly poverty level index
            FeatureColumn('Questionnaire', 'INDFMMPI', preproc_real, {
                          'cutoff': 4}),
            # Total savings/cash assets for the family
            FeatureColumn('Questionnaire', 'IND310',
                          preproc_real, {'cutoff': 6}),
            # How healthy is the diet
            FeatureColumn('Questionnaire', 'DBQ700', preproc_real, {
                          'cutoff': 6}),
            # of meals not home prepared
            FeatureColumn('Questionnaire', 'DBD895', preproc_real, {
                          'cutoff': 22}),
            # of meals from fast food or pizza place
            FeatureColumn('Questionnaire', 'DBD900',
                          preproc_real, {'cutoff': 22}),
            # Used nutrition info to choose fast foods
            FeatureColumn('Questionnaire', 'CBQ540',
                          preproc_real, {'cutoff': 3}),
            # Used nutrition info in restaurant
            FeatureColumn('Questionnaire', 'CBQ585', preproc_real, {
                          'cutoff': 3}),
            # Avg # alcoholic drinks/day - past 12 mos
            FeatureColumn('Questionnaire', 'ALQ130',
                          preproc_real, {'cutoff': 14}),
            # days have 4/5 drinks - past 12 mos
            FeatureColumn('Questionnaire', 'ALQ141Q', preproc_real, {
                          'cutoff': 22}),
            # Minutes walk/bicycle for transportation per day (average)
            FeatureColumn('Questionnaire', 'PAD645',
                          preproc_real, {'cutoff': 1200}),
            # Past wk # days cardiovascular exercise
            FeatureColumn('Questionnaire', 'PAQ677',
                          preproc_real, {'cutoff': 7}),
            # Ever told doctor had trouble sleeping?
            FeatureColumn('Questionnaire', 'SLQ050',
                          preproc_onehot, {'cutoff': 3}),
            # Cigarettes smoked in entire life
            FeatureColumn('Questionnaire', 'SMQ621', preproc_real, {
                          'cutoff': 8}),
            # How many days used an e-cigarette?
            FeatureColumn('Questionnaire', 'SMQ905', preproc_real, {
                          'cutoff': 30}),
            # In past week # days person smoked inside
            FeatureColumn('Questionnaire', 'SMD480',
                          preproc_real, {'cutoff': 7}),

            ### End Added Features ###

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
                          preproc_real, {'cutoff': 72})
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
