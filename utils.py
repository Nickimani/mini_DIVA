import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
import sys
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.linear_model import BayesianRidge
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from imputers.customMICEImputer import customMICEImputer
import sklearn.neighbors._base
import warnings

sys.modules["sklearn.neighbors.base"] = sklearn.neighbors._base
pd.options.mode.chained_assignment = None  # default='warn'
warnings.filterwarnings("ignore")

datapath = "./datasets/"


def read_dataset(dataset="iris", datapath=datapath):
    if dataset == "iris":
        data = pd.read_csv(datapath + "Iris.csv")
        data = data.dropna().reset_index(drop=True)
        Xinds = list(range(4))
        yind = 4  # set target variable

    elif dataset == "automobile":
        data = pd.read_csv(datapath + "Automobile.csv")
        data = data.dropna().reset_index(drop=True)
        Xinds = list(range(25))
        yind = 25

    elif dataset == "beijing":
        data = pd.read_csv(datapath + "Beijing_Multi-Site.csv")
        data = data.dropna().reset_index(drop=True)
        Xinds = list(range(1, 17))
        yind = 17

    elif dataset == "credit":
        data = pd.read_csv(datapath + "CreditApproval.csv")
        data = data.dropna().reset_index(drop=True)
        Xinds = list(range(15))
        yind = 15

    elif dataset == "dermatology":
        data = pd.read_csv(datapath + "Dermatology.csv")
        data = data.dropna().reset_index(drop=True)
        Xinds = list(range(1, 35))
        yind = 35

    elif dataset == "monks":
        # test dataset: classifier for categorical variables
        data = pd.read_csv(datapath + "Monks.csv")
        data = data.dropna().reset_index(drop=True)
        Xinds = list(range(1, 6))
        yind = 0

    elif dataset == "soybean":
        # test dataset: classifier for categorical variables
        data = pd.read_csv(datapath + "Soybean-large.csv")
        data = data.dropna().reset_index(drop=True)
        Xinds = list(range(1, 36))
        yind = 0

    elif dataset == "wine_red":
        # test dataset: classifier for categorical variables
        data = pd.read_csv(datapath + "Winequality-red.csv")
        data = data.dropna().reset_index(drop=True)
        Xinds = list(range(0, 11))
        yind = 11

    elif dataset == "wine_white":
        # test dataset: classifier for categorical variables
        data = pd.read_csv(datapath + "Winequality-white.csv")
        data = data.dropna().reset_index(drop=True)
        Xinds = list(range(0, 11))
        yind = 11

    elif dataset == "yeast":
        # test dataset: classifier for categorical variables
        data = pd.read_csv(datapath + "Yeast.csv")
        data = data.dropna().reset_index(drop=True)
        Xinds = list(range(0, 9))
        yind = 9

    X, y = data.iloc[:, Xinds], data.iloc[:, yind]

    return data, X, y, Xinds, yind


def iscategorical(x, threshold=0.12):
    """
    determine if x is a categorical variable.


    Inputs:
    ------------------------------------------------------------
    x: pd.DataFrame or np.ndarray, a vector


    Outputs:
    ------------------------------------------------------------
    Bool value
    """
    # convert x to np.ndarray
    if isinstance(x, pd.DataFrame):
        x = x.to_numpy()

    if x.dtype in ["object", "bool", "str"]:
        return True
    elif len(np.unique(x[~np.isnan(x)])) < threshold * len(
        x[~np.isnan(x)]
    ):  # this threshold may need to be changed
        return True
    else:
        return False


def get_num_cat_vars(df):
    """
    get the position of numerical and categorical variables

    Inputs:
    ------------------------------------------------------------
    df: pd.DataFrame

    Outputs:
    ------------------------------------------------------------
    num_vars: list, numerical variables
    cat_vars: list, categorical variables
    """

    num_vars = []
    cat_vars = []
    if not isinstance(df, pd.DataFrame):
        df = pd.DataFrame(df).infer_objects()
    for col in df.columns:
        if iscategorical(df[col]):
            cat_vars.append(col)
        else:
            num_vars.append(col)

    return num_vars, cat_vars


def set_fraction_missing(X, fraction=0.2, random_state=42):
    """
    Set a fraction of missing values in X, randomly in each feature

    Inputs:
    ------------------------------------------------------------
    X: pd.DataFrame or np.ndarray
    fraction: float, fraction of missing values
    random_state: int, random seed


    Output:
    ------------------------------------------------------------
    X_miss: pd.DataFrame or np.ndarray, X with missing values
    X_miss_idx: pd.DataFrame or np.ndarray, boolean matrix with True for missing values
    """

    X_miss = X.copy()
    for col in X.columns:
        X_miss.loc[:, col] = X_miss.loc[:, col].sample(
            frac=1 - fraction, random_state=random_state
        )
        random_state += 1
    X_miss_idx = X_miss.isnull()
    return X_miss, X_miss_idx


def mixerror(X_imp: np.ndarray, X: np.ndarray, X_miss_idx: np.ndarray, cat_vars=None):
    """
    Combined-error for both categorical and continuous variables,
    Reference:
    - Tang, F., & Ishwaran, H. (2017). Random forest missing data algorithms.
    Statistical Analysis and Data Mining: The ASA Data Science Journal, 10(6), 363-377


    Inputs:
    ------------------------------------------------------------
    X: np.ndarray
    X_true: np.ndarray
    X_miss_idx: where to compare
    cat_vars: bool vector, whether the variable is a categorical variable


    Output:
    ------------------------------------------------------------
    error: float
    """

    N, nvars = X.shape
    cat_error, num_error = 0, 0

    # identify categorical and continuous variables
    if cat_vars is None:
        num_features, cat_features = get_num_cat_vars(X)
    else:
        col_list = np.array(range(nvars))
        num_features, cat_features = (
            col_list[~np.array(cat_vars)],
            col_list[np.array(cat_vars)],
        )
    n_num, n_cat = len(num_features), len(cat_features)

    for num in num_features:
        if any(X_miss_idx[:, num]):
            var = np.var(X[X_miss_idx[:, num], num])
            if var <= 1e-6:
                var = 1
            num_error += np.sqrt(
                mean_squared_error(
                    X_imp[X_miss_idx[:, num], num], X[X_miss_idx[:, num], num]
                )
                / var
            )

    for cat in cat_features:
        if any(X_miss_idx[:, cat]):
            cat_error += np.sum(
                X_imp[X_miss_idx[:, cat], cat] != X[X_miss_idx[:, cat], cat]
            ) / np.sum(X_miss_idx[:, cat])

    error = 1 / max(1, n_cat) * cat_error + 1 / max(1, n_num) * (num_error)

    return error


def one_hot_encode(X, X_miss, num_vars, cat_vars):
    """
    Encode categorical features as a one-hot numeric array

    Inputs:
    ------------------------------------------------------------
    X: pd.DataFrame, original data without missing values
    X_miss: pd.DataFrame, X with missing values
    num_vars, cat_vars: list, column names of numerical/categorical variables


    Outputs:
    ------------------------------------------------------------
    X_: pd.DataFrame, one hot encoded X without missing values
    X_miss_: pd.DataFrame, one hot encoded X with missing values
    trans_names: ndarray of str objects, column names of one hot encoded variables (feature + "_" + str(category))
    ohe: one hot encoder object
    """
    X_cat = X[cat_vars]
    X_miss_idx = X_miss.isnull()

    ohe = ColumnTransformer(
        transformers=[
            (
                "OneHot",
                OneHotEncoder(sparse=False, drop="first", handle_unknown="error"),
                cat_vars,
            )
        ],
        remainder="passthrough",
    )
    X_trans_ = ohe.fit_transform(X_cat)
    trans_names = ohe.named_transformers_["OneHot"].get_feature_names_out()
    X_trans_df = pd.DataFrame(X_trans_, columns=trans_names)
    X_ = pd.concat([X[num_vars], X_trans_df], axis=1)

    X_trans = X_trans_df.copy()
    for col in cat_vars:
        miss = X_miss_idx[col]
        cols = X_trans.columns[X_trans.columns.str.contains(pat=col)]
        for x in cols:
            X_trans[x][miss] = np.nan

    X_miss_ = pd.concat([X_miss[num_vars], X_trans], axis=1)
    return X_, X_miss_, trans_names, ohe


def one_hot_decode(X, X_miss_, X_imp_, ohe, num_vars, cat_vars, trans_names):
    """
    Convert the one hot encoded data back to the original representation

    Inputs:
    ------------------------------------------------------------
    X: pd.DataFrame, original data without missing values
    X_miss_: pd.DataFrame, one hot encoded X with missing values
    num_vars, cat_vars: list, column names of numerical/categorical variables
    trans_names: list, column names of one hot encoded variables (feature + "_" + str(category))

    Output:
    ------------------------------------------------------------
    X_imp: pd.DataFrame, inverse transformed data
    """
    if not isinstance(X_imp_, pd.DataFrame):
        X_imp_ = pd.DataFrame(X_imp_, columns=X_miss_.columns)
    imp_num = X_imp_[num_vars]
    imp_cat = pd.DataFrame(
        ohe.named_transformers_["OneHot"].inverse_transform(X_imp_[trans_names]),
        columns=cat_vars,
    )

    X_imp = pd.concat([imp_num, imp_cat], axis=1).reindex(columns=X.columns)
    return X_imp


def impute(
    X_miss,
    X,
    y=None,
    train_idx=None,
    test_idx=None,
    encoding="label",
    method="mean",
    random_state=42,
):
    """
    impute missing values in X_miss using method.

    Inputs:
    ------------------------------------------------------------
    X_miss: pd.DataFrame, data with missing values
    X: pd.DataFrame, original
    train_idx,test_idx: training and testing set indices for the split
    encoding: {'label', 'one-hot'}, method for transforming categorical variables, default = 'label'
    method: str, imputation method

    Outputs:
    ------------------------------------------------------------
    X_imp: pd.DataFrame, imputed data
    X_le: pd.DataFrame, original data (categorical variables label encoded)
    """

    num_vars, cat_vars = get_num_cat_vars(X)

    X_le = X.copy()

    if train_idx is None or test_idx is None:
        train_idx, test_idx = X.index, [0]

    # use label encoder to transform categorical variables into integer space
    if len(cat_vars) > 0:
        le = LabelEncoder()
        for col in cat_vars:
            X_le[col] = le.fit_transform(X[col])
    X_miss_le = X_le.copy()

    # perform categorical encoding using one hot encoder or label encoder
    if encoding == "one-hot" and len(cat_vars) > 0:
        _, X_miss_le, trans_names, ohe = one_hot_encode(
            X_le, X_miss, num_vars, cat_vars
        )
        num_trans, cat_trans = list(X_miss_le.columns), []
    else:
        X_miss_le = X_le.copy()
        num_trans, cat_trans = num_vars.copy(), cat_vars.copy()
        X_miss_le[X_miss.isnull()] = np.nan

    X_imp = X_miss_le.copy()

    # impute missing values, depending on numerical or categorical variables
    imputers = {
        "zero": SimpleImputer(strategy="constant", fill_value=0),
        "mean": ColumnTransformer(
            transformers=[
                ("num_imp", SimpleImputer(strategy="mean"), num_trans),
                ("cat_imp", SimpleImputer(strategy="most_frequent"), cat_trans),
            ],
            verbose_feature_names_out=False,
        ),
        "median": ColumnTransformer(
            transformers=[
                ("num_imp", SimpleImputer(strategy="median"), num_trans),
                ("cat_imp", SimpleImputer(strategy="most_frequent"), cat_trans),
            ],
            verbose_feature_names_out=False,
        ),
        "mode": SimpleImputer(strategy="most_frequent"),
        "MICE": IterativeImputer(random_state=random_state, estimator=BayesianRidge()),
        "customMICE": customMICEImputer(
            num_vars,
            cat_vars,
            threshold=0,
            iteration=2,
            methodN="ridge",
            methodC="random_forest",
            solver="newton-cg",
        ),
    }

    # pre-define fit-transform hyperparameters here
    fit_params = {"zero": {}, "mean": {}, "median": {}, "mode": {}, "MICE": {}}
    trans_params = {"zero": {}, "mean": {}, "median": {}, "mode": {}, "MICE": {}}
    imputer = imputers[method]

    # fit transform on given data
    if method == "customMICE":
        X_imp.iloc[train_idx] = imputer.fit(X_miss_le.iloc[train_idx])
        X_imp.iloc[test_idx] = imputer.transform(X_miss_le.iloc[test_idx])
    elif method in ["zero", "mode", "MICE"]:
        X_imp.iloc[train_idx] = imputer.fit_transform(
            X_miss_le.iloc[train_idx], y.iloc[train_idx], **fit_params[method]
        )
        X_imp.iloc[test_idx] = imputer.transform(
            X_miss_le.iloc[test_idx], **trans_params[method]
        )
    elif method in ["mean", "median"]:
        train_imp = pd.DataFrame(
            imputer.fit_transform(X_miss_le.iloc[train_idx]),
            columns=imputer.get_feature_names_out(),
        ).reindex(columns=X_imp.columns)
        test_imp = pd.DataFrame(
            imputer.transform(X_miss_le.iloc[test_idx]),
            columns=imputer.get_feature_names_out(),
        ).reindex(columns=X_imp.columns)
        X_imp.iloc[train_idx], X_imp.iloc[test_idx] = train_imp, test_imp
    else:
        raise ValueError(
            "method must be one of: zero, mean, median, mode, MICE, customMICE."
        )

    # one hot decoding
    if encoding == "one-hot" and len(cat_vars) > 0:
        X_imp = one_hot_decode(
            X, X_miss_le, X_imp, ohe, num_vars, cat_vars, trans_names
        )

    # transform back to pd.DataFrame
    if not isinstance(X_imp, pd.DataFrame):
        X_imp = pd.DataFrame(X_imp, columns=X_miss_le.columns)

    return X_imp, X_le


def evaluate_imputation(X, X_imp, X_true, X_miss_idx, error_method="mix"):
    """
    Input:
    ------------------------------------------------------------
    X: X original data
    X_imp: X data with missing values imputed 
    X_true: 
    X_miss_idx:
    error_method:
    
    Output:
    error: Error introduced to the model after imputation of missing values
    """
    # use original data X to get numerical or categorical variables
    num_vars, cat_vars = get_num_cat_vars(X)
    error = 0

    if error_method == "mse" or error_method == "rmse":
        # calculate mse for numerical variables
        for num in num_vars:
            if any(X_miss_idx[num]):
                error += (
                    1
                    / len(num_vars)
                    * mean_squared_error(
                        X_imp[num][X_miss_idx[num]], X_true[num][X_miss_idx[num]]
                    )
                )

        # calculate proportion error for categorical variables
        for cat in cat_vars:
            if any(X_miss_idx[cat]):
                error += (
                    1
                    / len(cat_vars)
                    * np.sum(
                        X_imp[cat][X_miss_idx[cat]] != X_true[cat][X_miss_idx[cat]]
                    )
                    / np.sum(X_miss_idx[cat])
                )

        if error_method == "rmse":
            error = np.sqrt(error)

    elif error_method == "mix":
        cat_features = [x in cat_vars for x in X.columns]
        error = mixerror(
            X_imp.to_numpy(),
            X_true.to_numpy(),
            X_miss_idx.to_numpy(),
            cat_vars=cat_features,
        )

    return error


def train_impute_classifier(
    X,
    y,
    encoding="label",
    impute_method="mean",
    error_method="mix",
    impute_fraction=0.2,
    random_state=42,
):
    """
    Inputs: 
    ------------------------------------------------------------
    X: pd.DataFrame, X original data,
    y: pd.Series, y original data
    encoding: str, Encoding method to use on categorical data; label, one-hot
    impute_methods: str, Imputation method for missing values; zero, median, mode, mean, MICE, CustomMICE
    error_method: str, Error method to use; mse, rmse, mix
    impute_fraction: float, Portion of dataset to be changed to missing value representation
    random_state: int, 
    
    Outputs:
    ------------------------------------------------------------
    acc: Model accuracy when trained on imputed data
    error: Error introduced to the data due to imputation
    X_miss: X data with fraction of missing values 
    X_miss_idx: Indexes of records with missing values
    X_imp: X data with missing values that have been imputed
    X_le: Original X data after label encoding the categorical data
    """
    # set missing values
    X_miss, X_miss_idx = set_fraction_missing(
        X, fraction=impute_fraction, random_state=random_state
    )

    # impute missing values
    if iscategorical(y):
        splitter = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)
    else:
        splitter = KFold(n_splits=5, shuffle=True, random_state=random_state)

    acc, error = [0] * 5, [0] * 5
    _, cat_vars = get_num_cat_vars(X)

    for i, (train_idx, test_idx) in enumerate(splitter.split(X, y)):
        (
            X_imp,
            X_le,
        ) = impute(
            X_miss,
            X,
            y,
            train_idx,
            test_idx,
            encoding=encoding,
            method=impute_method,
            random_state=random_state,
        )

        error[i] = evaluate_imputation(
            X,
            X_imp.iloc[test_idx],
            X_le.iloc[test_idx],
            X_miss_idx.iloc[test_idx],
            error_method=error_method,
        )

        # split data into train and test sets
        X_train, X_test, y_train, y_test = (
            X_imp.iloc[train_idx],
            X_imp.iloc[test_idx],
            y.iloc[train_idx],
            y.iloc[test_idx],
        )

        # train random forest classifier
        mdl = (
            RandomForestClassifier(random_state=42)
            if iscategorical(y)
            else RandomForestRegressor(random_state=42)
        )
        if encoding == "one-hot":
            ohe = ColumnTransformer(
                transformers=[
                    (
                        "OneHot",
                        OneHotEncoder(
                            sparse=False, drop="first", handle_unknown="ignore"
                        ),
                        cat_vars,
                    )
                ],
                remainder="passthrough",
            )
            rf = Pipeline(steps=[("one-hot", ohe), ("model", mdl)])
        else:
            rf = Pipeline(steps=[("model", mdl)])

        rf.fit(X_train, y_train)

        # calculate accuracy of classifier on test set/RMSE of regressor on test set
        if iscategorical(y):
            acc[i] = rf.score(X_test, y_test)  # accuracy
        else:
            y_pred = rf.predict(X_test)
            # acc[i] = mean_squared_error(np.array(y_test),y_pred,squared = False)# RMSE
            acc[i] = 1 - mean_absolute_percentage_error(
                np.array(y_test), y_pred
            )  # MAPE

    return acc, error, X_miss, X_miss_idx, X_imp, X_le
