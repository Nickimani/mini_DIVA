import numpy as np
import pandas as pd
import copy
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor

class hotDeckImputer:

    """
    Imputation method: Hot Deck imputation method)

    Parameters
    --------------------------------------------
    numericalCols : list
        Indices of the Numerical Columns

    categoricalCols : list
        Indices of the Categorical Columns

    n_neighbors: int
        Number of neighbors the imputer should use in creating a 'hot deck'


    Attributes
    --------------------------------------------
    numericalColumns : list
        Names of the Numerical Columns

    categoricalColumns : list
        Names of the Categorical Columns

    numericalIndices : list
        Indices of the Numerical Columns

    categoricalIndices : list
        Indices of the Categorical Columns
    
    columns_ : Pandas Series
        A series of the headers of the Pandas Dataframe dataset

    neighbors_: int
        Number of neighbors the imputer should use in creating a 'hot deck'

    imputedValList : list
        A list containing all imputed values (means or modes), each value for each column of the dataset

    modelDict : list
        A list containing all savedModels, each model for each column of the dataset
    
    originalData : list
        It is a list of lists, converted from the Pandas Dataframe dataset
        The original list is stored internally before any following operations

    intermediateData : list
        It is a list of lists. 
        The original data after being imputed with the mean/mode of the respective column


    Tips
    --------------------------------------------
    Both the test and train data have missing values therefore all sets of data need to be run through the _impute function.

    
    Example
    --------------------------------------------
    hot_deck = hotDeckImputer(numerical_cols, categorical_cols, n_neighbors)
    outputTrain = hot_deck.fit(df_nanTrain)
    outputTest = hot_deck.transform(df_nanTest)

    """

    def __init__(
            self, 
            numericalColumns: list,
            categoricalColumns: list,
             n_neighbors: int
     ):
        self.numericalColumns = numericalColumns
        self.categoricalColumns = categoricalColumns
        self.neighbors_ = n_neighbors


    def _impute(self, matrix: np.ndarray, target_col: int, train: bool):
        """
        Impute the missing values in the dataset with mean for the numerical columns and mode for categorical columns.

        Inputs:
        ------------------------------------------------------------
        matrix: np.array, Sparse matrix 
        target_col: int, Index of column to be imputed
        train: bool, Whether it is running in the training phase or not

        Outputs:
        ------------------------------------------------------------
        matrix: np.array, Matrix with the target column imputed with either mean or mode
        """
        # obtain imputing values from the non-null records in the data fed in the training phase
        if train:            
            if target_col in self.categoricalIndices:
                freqCountDict = {}
                for row in range(len(matrix)):
                    record = matrix[row][target_col]
                    if not np.isnan(record):
                        freqCountDict[record] = (freqCountDict.get(record, 0) + 1)
                value = max(freqCountDict, key=freqCountDict.get)
                
            else:
                nonNullValues = []
                for row in range(len(matrix)):
                    record = matrix[row][target_col]
                    if not np.isnan(record):
                        nonNullValues.append(record)
                value = sum(nonNullValues) / len(nonNullValues)
            
            self.imputedValList.append(value)

        # when testing, get the value from the saved imputed value list created during training 
        else:
            value = self.imputedValList.pop(0)

        # fill in the missing values
        for row in range(len(matrix)):
            if np.isnan(matrix[row][target_col]) or matrix[row][target_col] == None:
                matrix[row][target_col] = value

        return matrix
            


    def _runModel(self, matrix: np.ndarray, target_col: int, train: bool):
        """
        Runs a regressor using target_col as the target variable.
        
        Inputs:
        ------------------------------------------------------------
        matrix: np.array, A matrix of imputed variables except the target variable
        target_col: int, Index of the target variable,
        train: bool, Whether it is the training phase or not

        Outputs:
        ------------------------------------------------------------
        matrix: np.array, Matrix with target column imputed by a regressor
        """
        matrixCopy = copy.deepcopy(matrix)
        targetMatrix = [matrixCopy[row].pop(target_col) for row in range(len(matrix))]

        trainFeatures, trainTarget = [], []
        testFeatures, testTarget = [], []
            
        for feature, target in zip(matrixCopy, targetMatrix):
            if target is not None:
                # making sure target variable records are integers to prevent errors when training the model
                if target_col in self.categoricalIndices:
                    trainFeatures.append(feature)
                    trainTarget.append(round(target))
                else:
                    trainFeatures.append(feature)
                    trainTarget.append(target)

            elif target is None:
                testFeatures.append(feature)
                testTarget.append(target)   

        # print(f"""
        #       Target Column is: {self.columns_[target_col].title()}
        #       {trainTarget} \n 
        #       {trainFeatures} \n 
        #       {testFeatures} \n
        #       {testTarget}
        #       """)     
        
        # use classifier if categorical target column else use the regressor
        if train:
            if target_col in self.categoricalIndices:
                model = KNeighborsClassifier(n_neighbors=self.neighbors_, weights='distance', n_jobs=-1)
                model.fit(trainFeatures, trainTarget)

            elif target_col in self.numericalIndices:
                model = KNeighborsRegressor(n_neighbors=self.neighbors_, weights='distance', n_jobs=-1)
                model.fit(trainFeatures, trainTarget)
            
            # all columns in the dataset traines have their respective models
            # however not always will you find that the target column has missing values
            # in a case where there are no missing values in the target, there will be no testFeatures
            # thus these conditions prevent the model throwing an error in such a case
            if len(testFeatures) > 0:
                predictions = model.predict(testFeatures).tolist()    
            else:
                predictions = []

            # adding trained model to dictionary of available models for testing
            self.modelDict[target_col] = model
        
        else:
            # print(self.modelDict)
            model = self.modelDict[target_col]
            # same condition as used above
            if len(testFeatures) > 0:
                predictions = model.predict(testFeatures).tolist()
            else:
                predictions = []

        # perform the imputation with values got from the hotDeck
        while len(predictions) > 0:
            for row in range(len(matrix)):
                if matrix[row][target_col] is None:
                    matrix[row][target_col] = predictions.pop(0)

        return matrix

        

    def _runImputations(self, dataframe: pd.DataFrame, train: bool):
        """
        Runs all iterations of null value imputation in the data. The number of iterations are decided by then number of columns in the data.\
        
        Inputs:
        ------------------------------------------------------------
        dataframe: pd.DataFrame, The data to be used in either training or testing the imputer
        train: bool, Whether it is the training phase or not

        Outputs:
        ------------------------------------------------------------
        dataframe: pd.DataFrame, Dataframe with all null values imputed with the hot deck method
        """
        self.columns_ = dataframe.columns.to_list()
        matrix = dataframe.values.tolist()
        self.originalData = copy.deepcopy(matrix)    
        
        # separate numerical and categorical indexes
        cat_indexes = []
        num_indexes = []        
        for idx, col  in enumerate(self.columns_):            
            if col in self.categoricalColumns:
                cat_indexes.append(idx)
            else:
                num_indexes.append(idx)

        self.categoricalIndices = cat_indexes
        self.numericalIndices = num_indexes
        
        # create a mask for Null records
        nullRecordsMask = {idx: list() for idx in range(len(matrix[0]))}

        for col in range(len(matrix[0])):
            for row in range(len(matrix)):
                if np.isnan(matrix[row][col]) or matrix[row][col] == None:
                    nullRecordsMask[col].append(row)

        # getting rid of columns with no missing values
        nullRecordsMask = {idx: val for idx, val in nullRecordsMask.items() if len(val) != 0}

        # print(nullRecordsMask, "\n")
        # run the mean/mode imputation
        for col in range(len(matrix[0])):
            matrix = self._impute(matrix, col, train)

        self.intermediateData = copy.deepcopy(matrix)

        # make sure null records are represented as None before running the model imputation
        for target_col in range(len(matrix[0])):
            # since it does not matter whether it is training phase or not when it comes to model imputation
            # it does not matter because both train and test data fed to the imputer have missing values
            if target_col in nullRecordsMask.keys():
                nullIndices = nullRecordsMask[target_col]
                for idx in nullIndices:
                    matrix[idx][target_col] = None
                matrix = self._runModel(matrix, target_col, train)

            else:
                matrix = self._runModel(matrix, target_col, train)
            
            
        df = pd.DataFrame(matrix, columns=self.columns_)

        return df
    


    def fit(self, dataframe):
        """
        The fit method to be used in the train set. It uses the _runImputations method keeping train=True

        Input:
        ------------------------------------------------------------
            dataframe: Input Pandas DataFrame of train set

        Output:
        ------------------------------------------------------------
            dataframe: The imputed DataFrame for the train set
        """
        self.imputedValList = []
        self.modelDict = {}

        df = self._runImputations(dataframe,
                                 train=True
                                 )
        
        return df
    
    def transform(self, dataframe):
        """
        The transform method to be used in the test set. It uses the _runImputations method keeping train=False
        
        Input:
        ------------------------------------------------------------
            dataframe: Input Pandas DataFrame of test set
        
        Output:
        ------------------------------------------------------------
            dataframe: The imputed DataFrame for the test set
        """
        df = self._runImputations(dataframe,
                                 train=False
                                 )
        
        return df
    