import pandas as pd
import numpy as np
import copy
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso
from sklearn.linear_model import Ridge
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier


class customMICEImputer:

    """
    Imputation method Multivariate Imputation By Chained Equations (MICE)

    .. versionadded:: 0.0.1

    .. note::
        Run method mice.fit on the training dataset and mice.transform on the test dataset
        These methods returns imputed Pandas Dataframes

    Parameters
    -----------
    numericalCols : list
        Indices of the Numerical Columns

    categoricalCols : list
        Indices of the Categorical Columns


    Attributes
    -----------
    solver : str
        The solver for Logistic Regression. It only works when methodC = 'logistic'
        Possible options 'sags, 'saga', 'newton-cg', 'liblinear', 'lbfgs', etc.

    savedModel : object
        The Regression model is saved internally after it is fit on the trainning data
        The model is later used in predicting for the test data

    savedModelsList : list
        A list containing all savedModels, each model for each column of the dataset

    savedImputedValList : list
        A list containing all imputed values, each value for each column of the dataset

    headers : Pandas Series
        A series of the headers of the Pandas Dataframe dataset

    datasetOriginal : list
        It is a list of lists, converted from the Pandas Dataframe dataset
        The original list is stored internally before any following operations

    iteration : int
        Max number of iterations

    methodN : str
        The Regression method for Numerical entries
        Choose from linear, lasso or ridge

    methodC : str
        The Regression method for Categorical entries
        Choose from 'logistic' (with solver= 'sags, 'saga', 'newton-cg', 'liblinear', 'lbfgs', etc.) and 'random_forest'


    Tips
    ----
    For Dataset with only Numerical columns, try different iterations starting from 2. It is computationally less expensive.
    Try different methodN.

    For Dataset with mixed entries or only Categorical columns, as it can be computationally expensive,
    try iteration=2 only if it is too much time consuming.
    Random Forest regressor is usually less expensive and more efficient.

    For small Dataset while trying methodC='logistic', use solver='liblinear'. But it is limited to one-versus-rest schemes.

    For multiclass problems, only 'newton-cg', 'sag', 'saga' and 'lbfgs' handle multinomial loss.

    Example
    -------
    mice = customMICEImputer(yind, numerical_cols, categorical_cols, iteration=25, methodN='ridge', methodC='random_forest')
    outputTrain = mice.fit(df_nanTrain)
    outputTest = mice.transform(df_nanTest)

    """

    def __init__(
        self,
        numericalCols,
        categoricalCols,
        threshold=0,
        iteration=2,
        methodN="linear",
        methodC="logistic",
        solver="sag",
    ):
        self.numericalCols = numericalCols
        self.categoricalCols = categoricalCols
        self.threshold = threshold
        self.iteration = iteration
        self.methodN = methodN
        self.methodC = methodC
        self.solver = solver

    def _impute(self, matrix, col, train, method="mean"):
        """
        Impute missing values in a column of the matrix using the specified imputation method.
        For Numerical columns, when using this method, imputation method = 'mean'. Else, imputation method = 'mode'.
        This method is used in _run method.
        Input:
            matrix: sparse matrix
            col: column 'col' to be imputed
            method: imputation method
        Output:
            matrix: matrix with column 'col' imputed
        """
        if train:
            if method == "mean":
                nonNullValues = []
                for i in range(len(matrix)):
                    if not np.isnan(matrix[i][col]):
                        nonNullValues.append(matrix[i][col])

                if len(nonNullValues) > 0:
                    val = sum(nonNullValues) / len(nonNullValues)

            elif method == "mode":
                occurrenceDict = {}
                for i in range(len(matrix)):
                    if not np.isnan(matrix[i][col]):
                        occurrenceDict[matrix[i][col]] = (
                            occurrenceDict.get(matrix[i][col], 0) + 1
                        )
                val = max(occurrenceDict, key=occurrenceDict.get)

            else:
                raise ValueError("Invalid attribute value.")

            self.savedImputedVal = val

        else:
            val = self.savedImputedValList.pop(0)

        # if val:
        for i in range(len(matrix)):
            if np.isnan(matrix[i][col]):
                matrix[i][col] = val

        return matrix

    def _regression(self, matrix, targetCol, method, train):
        """
        Perform regression-based imputation for a target column in the matrix using the specified regression method.
        For Numerical columns, regression methods: linear, lasso, ridge
        For Categorical columns, regression methods: logistic, random forest
        Function:
            Step 1: Regression model is fitted for rows where entries in target column are not nan. Corresponding entries of feature columns are taken as features (this step is done if train=True)
            Step 2: Regression model is used to transform/predict the nan entries in target column keeping corresponding entries of feature columns as features
        Input:
            matrix: Matrix with all columns except target column 'targetCol' imputed with _impute method
            targetCol: The target column for the regression method
            method: Regression method = 'linear', 'lasso', 'ridge', 'logistic' or 'random_forest'
            train: A bool denoting if it is the training phase or not
        Output:
            matrix: Output transformed/predicted matrix
        """
        matrixFeature = copy.deepcopy(matrix)

        matrixTarget = [matrixFeature[row].pop(targetCol) for row in range(len(matrix))]

        trainListFeatures, trainListTarget = [], []
        testListFeatures = []
        for valFeatures, valTarget in zip(matrixFeature, matrixTarget):
            if valTarget is not None:
                trainListFeatures.append(valFeatures)
                trainListTarget.append(valTarget)
            else:
                testListFeatures.append(valFeatures)

        if train:
            if method == "linear":
                model = LinearRegression()
                model.fit(trainListFeatures, trainListTarget)
                predictionsList = list(model.predict(testListFeatures))
            elif method == "lasso":
                model = Lasso(alpha=1.0)
                model.fit(trainListFeatures, trainListTarget)
                predictionsList = list(model.predict(testListFeatures))
            elif method == "ridge":
                model = Ridge(alpha=1.0)
                model.fit(trainListFeatures, trainListTarget)
                predictionsList = list(model.predict(testListFeatures))
            elif method == "logistic":
                model = LogisticRegression(
                    multi_class="multinomial",
                    solver=self.solver,
                )
                try:
                    model.fit(trainListFeatures, trainListTarget)
                    predictionsList = list(model.predict(testListFeatures))
                except:  # for the case when there is only 1 class
                    val = trainListTarget[0]
                    model = val
                    predictionsList = [val] * len(testListFeatures)
            elif method == "random_forest":
                model = RandomForestClassifier()
                model.fit(trainListFeatures, trainListTarget)
                predictionsList = list(model.predict(testListFeatures))
            else:
                raise ValueError("Invalid attribute value.")

            self.savedModel = model
        else:
            model = self.savedModelsList.pop(0)
            try:
                predictionsList = list(model.predict(testListFeatures))
            except:  # for the case when there is only 1 class
                predictionsList = [model] * len(testListFeatures)

        while len(predictionsList) > 0:
            for row in range(len(matrix)):
                if matrix[row][targetCol] is None:
                    valueToImpute = predictionsList.pop(0)
                    matrix[row][targetCol] = valueToImpute

        return matrix

    def _run(self, matrix, noneIndices, methodN, methodC, categoricalCols, train):
        """
        This method is one single iteration of MICE
        Function:
            Step 1: _impute method is used to impute all columns of the matrix except the first column
            Step 2: _regression method is used to fit and transform on the first column as target column
            Step 3: Go back to step 1 and do it for the second column, and so on.
            Step 4: Fitted regression models are saved as class attributes (if train=True)
        Input:
            matrix: The input matrix which is sparsed for 1st iteration, or not sparsed for next iterations
            noneIndices: A dictionary denoting which indices of the original matrix are nan
            methodN: The regression method for numerical columns
            methodC: The regression method for categorical columns
            categoricalCols: A list of the names of the categorical columns
            train: A bool denoting if it is the training phase or not
        Output:
            matrix: The final imputed matrix (after applying _impute as well as _regression methods)
            prevMatrix: The intermediate matrix (only after _impute method)
        """
        categoricalColsIdx = []
        for idx, val in enumerate(self.headers):
            if val in categoricalCols:
                categoricalColsIdx.append(idx)

        for col in range(len(matrix[0])):
            if col in categoricalColsIdx:
                matrix = self._impute(matrix, col, train=train, method="mode")
            else:
                matrix = self._impute(matrix, col, train=train, method="mean")

            if train:
                self.savedImputedValList.append(self.savedImputedVal)

        prevMatrix = copy.deepcopy(matrix)

        for targetCol in range(len(matrix[0])):
            if targetCol in list(noneIndices.keys()):
                noneRowList = noneIndices[targetCol]
                for noneRow in noneRowList:
                    matrix[noneRow][targetCol] = None

            if targetCol in categoricalColsIdx:
                matrix = self._regression(
                    matrix, targetCol, method=methodC, train=train
                )
            else:
                matrix = self._regression(
                    matrix, targetCol, method=methodN, train=train
                )

            if train:
                self.savedModelsList.append(self.savedModel)

        return matrix, prevMatrix

    def subtractMatrices(self, matrix1, matrix2):
        """
        Subtract two matrices element-wise and return the result.
        Input:
            matrix1, matrix2: The 2 input matrices
        Output:
            subtractedMatrix: The output matrix after the element-wise subtraction
        """
        subtractedMatrix = [
            [0 for _ in range(len(matrix1[0]))] for _ in range(len(matrix1))
        ]
        for i in range(len(matrix1)):
            for j in range(len(matrix1[0])):
                subtractedMatrix[i][j] = matrix1[i][j] - matrix2[i][j]

        return subtractedMatrix

    def squaredSummedSubtractedMatrix(self, matrix):
        """
        Compute the squared sum of all elements in a matrix.
        Input:
            matrix: Input matrix
        Output:
            squareSummed: Squared sum of the elements of the input matrix
        """
        squareSummed = 0
        for row in range(len(matrix)):
            for col in range(len(matrix[0])):
                squareSummed += matrix[row][col] ** 2

        return squareSummed

    def _main(self, df, threshold, iteration, methodN, methodC, solver, train):
        """
        The multiple iteration part that uses _run method multiple times as multiple iterations
        Input:
            df: Input Pandas DataFrame (transformed to matrix in this method)
            threshold: The threshold of difference between matrix and prevMatrix (from _run method) below which iteration is stopped
            iteration: The max number of iterations
            methodN: The regression method for numerical columns
            methodC: The regression method for categorical columns
            solver: The solver for when methodC = 'logistic'
            train: A bool denoting if it is the training phase or not
        Output:
            df: The imputed DataFrame
        """

        # convert to list of lists
        headers = df.columns.values.tolist()
        self.headers = headers
        dataset = df.values.tolist()

        # store the original dataset
        self.datasetOriginal = copy.deepcopy(dataset)

        # main part
        noneIndices = {}
        for col in range(len(dataset[0])):
            for row in range(len(dataset)):
                if np.isnan(dataset[row][col]):
                    try:
                        rowList = noneIndices[col]
                        rowList.append(row)
                        noneIndices[col] = rowList
                    except:
                        noneIndices[col] = [row]

        self.solver = solver

        for _ in range(iteration):
            matrix, prevMatrix = self._run(
                dataset, noneIndices, methodN, methodC, self.categoricalCols, train
            )
            subtractedMatrix = self.subtractMatrices(matrix, prevMatrix)
            squareSummed = self.squaredSummedSubtractedMatrix(subtractedMatrix)

            if squareSummed <= threshold:
                df = pd.DataFrame(matrix, columns=headers)
                return df
        df = pd.DataFrame(matrix, columns=headers)
        return df

    def fit(self, df):
        """
        The fit method to be used in the train set. It uses the _main method keeping train=True
        Input:
            df: Input Pandas DataFrame of train set
        Output:
            df: The imputed DataFrame for the train set
        """

        self.savedModelsList = []
        self.savedImputedValList = []
        df = self._main(
            df,
            self.threshold,
            self.iteration,
            self.methodN,
            self.methodC,
            self.solver,
            train=True,
        )
        return df

    def transform(self, df):
        """
        The transform method to be used in the test set. It uses the _main method keeping train=False
        Input:
            df: Input Pandas DataFrame of test set
        Output:
            df: The imputed DataFrame for the test set
        """

        df = self._main(
            df,
            self.threshold,
            self.iteration,
            self.methodN,
            self.methodC,
            self.solver,
            train=False,
        )
        return df
