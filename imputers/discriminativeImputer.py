import numpy as np
import pandas as pd
import copy
import sys
import subprocess

# ensure that tensorflow is installed, if not install it
subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'tensorflow'])
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Dense, Dropout
from tensorflow.keras import losses, metrics

class discriminativeDLImputer:

    """
    Imputation method: Hot Deck imputation method)

    Parameters
    --------------------------------------------
    numericalCols : list
        Indices of the Numerical Columns

    categoricalCols : list
        Indices of the Categorical Columns


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

    imputedValList : list
        A list containing all imputed values (means or modes), each value for each column of the dataset

    modelDict : Dict
        A dictionary containing all savedModels, a model for each column in the dataset
    
    originalData : list
        A list of lists, converted from the original Pandas Dataframe dataset
        The original list is stored internally before any following operations

    intermediateData : list
        A list of lists. 
        The original data after being imputed with the mean/mode of the respective column


    Tips
    --------------------------------------------
    Both the test and train data have missing values therefore all sets of data need to be run through the _impute function.

    
    Example
    --------------------------------------------
    hot_deck = hotDeckImputer(numerical_cols, categorical_cols)
    outputTrain = hot_deck.fit(df_nanTrain)
    outputTest = hot_deck.transform(df_nanTest)

    """

    def __init__(
            self, 
            numericalColumns: list,
            categoricalColumns: list,
     ):
        self.numericalColumns = numericalColumns
        self.categoricalColumns = categoricalColumns

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
        testFeatures = []
            
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
        
        
        # use classifier if categorical target column else use the regressor
        if train:
            if target_col in self.categoricalIndices:
                n_categories = len(np.unique(trainTarget))
                if n_categories <= 2:            
                    # instantiate the sequential model
                    model = Sequential()
                    # add the neccessary input, hidden, dropout and output layers
                    model.add(Input(shape=(len(trainFeatures[0]), )))
                    model.add(Dense(56, activation='relu'))
                    model.add(Dropout(0.2))
                    model.add(Dense(24, activation='relu'))
                    model.add(Dropout(0.05))
                    model.add(Dense(4, activation="sigmoid"))
                    model.add(Dense(1))
                    # compile the model, fit and use it for predictions
                    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=[metrics.CategoricalAccuracy()])
                    model.fit(np.asarray(trainFeatures).astype(np.float32), np.asarray(trainTarget).astype(np.float32), epochs=100, verbose=0, batch_size=len(trainFeatures[0]), validation_split=0.2)
                    # all columns in the dataset trained should have their respective models
                    # however not always will you find that the target column has missing values
                    # in a case where there are no missing values in the target, there will be no testFeatures
                    # thus these conditions prevent the model throwing an error in such a case
                    if len(testFeatures) > 0:
                        predictions = model.predict(np.asarray(testFeatures).astype(np.float32))
                        predictions = [val*-1 if val < 0 else val for _ in predictions.tolist() for val in _]  # converting negative values in prediction to positive
                        predictions = [1 if val > 1 else val for val in predictions]  # making sure there are no values greater than 1 since it should be binary
                        predictions = [1 if val > 0.5 else 0 for val in predictions]  # rounding off the predictions to get either 0 or 1
                    else:
                        predictions = []

                else:
                    # instantiate the sequential model
                    model = Sequential()
                    # add the neccessary input, hidden, dropout and output layers
                    model.add(Input(shape=(len(trainFeatures[0]), )))
                    model.add(Dense(56, activation='relu'))
                    model.add(Dropout(0.2))
                    model.add(Dense(24, activation='relu'))
                    model.add(Dropout(0.05))
                    model.add(Dense(4, activation="softmax"))
                    model.add(Dense(1))
                    # compile the model, fit and use it for predictions
                    model.compile(loss=losses.CategoricalCrossentropy(), optimizer='adam', metrics=[metrics.CategoricalAccuracy()])
                    model.fit(np.asarray(trainFeatures).astype(np.float32), np.asarray(trainTarget).astype(np.float32), epochs=100, verbose=0, batch_size=len(trainFeatures[0]), validation_split=0.2)
                    if len(testFeatures) > 0:
                        predictions = model.predict(np.asarray(testFeatures).astype(np.float32))
                        predictions = [val*-1 if val < 0 else val for _ in predictions.tolist() for val in _]
                        predictions = [round(val) for val in predictions]  # rounding off to get the nearest class
                    else:
                        predictions = []


            elif target_col in self.numericalIndices:
                model = Sequential()                
                model.add(Input(shape=(len(trainFeatures[0]), )))  # input layer
                model.add(Dense(56, activation='relu'))  # hidden layer
                model.add(Dropout(0.2))
                model.add(Dense(24, activation='relu'))  # hidden layer
                model.add(Dropout(0.05))
                model.add(Dense(4, activation="relu"))  # hidden layer
                model.add(Dense(1))  # output layer
                model.compile(loss=losses.MeanSquaredError(), optimizer='adam', metrics=[metrics.MeanSquaredError()])
                model.fit(np.asarray(trainFeatures).astype(np.float32), np.asarray(trainTarget).astype(np.float32), epochs=100, verbose=0, batch_size=len(trainFeatures[0]), validation_split=0.2)
                if len(testFeatures) > 0:
                    predictions = model.predict(np.asarray(testFeatures).astype(np.float32))
                    predictions = [round(val, 2) for rec in predictions.tolist() for val in rec]
                else:
                    predictions = []

            self.modelDict[target_col] = model

        else:
            model = self.modelDict[target_col]
            if len(testFeatures) > 0:
                if target_col in self.categoricalColumns:                    
                    if n_categories <= 2:
                        predictions = model.predict(np.asarray(testFeatures).astype(np.float32))
                        predictions = [val*-1 if val < 0 else val for _ in predictions.tolist() for val in _]  # converting negative values in prediction to positive
                        predictions = [1 if val > 1 else val for val in predictions]  # making sure there are no values greater than 1 since it should be binary
                        predictions = [1 if val > 0.5 else 0 for val in predictions]  # rounding off the predictions to get either 0 or 1
                    else:
                        predictions = model.predict(np.asarray(testFeatures).astype(np.float32))
                        predictions = [val*-1 if val < 0 else val for _ in predictions.tolist() for val in _]
                        predictions = [round(val) for val in predictions]  # rounding off to get the nearest class
                else:
                    predictions = model.predict(np.asarray(testFeatures).astype(np.float32))
                    predictions = [val for rec in predictions.tolist() for val in rec]
                
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
                if np.isnan(matrix[row][col]):
                    nullRecordsMask[col].append(row)
        
        # run the mean/mode imputation
        for col in range(len(matrix[0])):
            matrix = self._impute(matrix, col, train)

        self.intermediateData = copy.deepcopy(matrix)

        # make sure null records are represented as None before running the model imputation
        for target_col in range(len(matrix[0])):
            # since it does not matter whether it is training phase or not when it comes to model imputation
            # it does not matter because both train and test data fed to the imputer have missing values
            nullIndices = nullRecordsMask[target_col]
            for idx in nullIndices:
                matrix[idx][target_col] = None

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
    