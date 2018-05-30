import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split


class Perceptron():
    """This class represents the Perceptron Algorithm.
    I implemented this class as my beginning to Machine Learning study.
    The algorithm is good for classification problem with two classes.

    * treshold = (int) a treshold for perceptron
    * learning_rate = (float) small number close to 0
    * target_name = (str) name of column with the target
    * test_size = (float/int/None) If float, should be between 0.0 and 1.0,
                    and represent the proportion of the dataset to include in the test split. 
                    If int, represents the absolute number of test samples. 
                    If None, the value is set to the complement of the train size. 
                    By default, the value is set to 0.25. 
    * file_path = (str) path to the data
    * sep = (str) how to split data in file; default: " "
    * header = (int) no. of row with names of columns; default None
    * index_col = (int) no. of column with index values; default None
    * columns = (list) if there is no header in data, set the list with colomns names; default None 


    """

    def __init__(self, treshold, learning_rate, target_name, test_size, file_path, sep = " ", header = None, index_col = None, columns = None):
        """Inicialize the perceptron""" 
        self._test_size = test_size
        self.target_name = target_name
        self.treshold = treshold
        self.learning_rate = learning_rate

        self._dataset = pd.read_csv(file_path, sep = sep, header = header, index_col = index_col)
        if columns is not None:
            self._dataset.columns = columns
        self.train_set,self.test_set, self.train_target, self.test_target = Perceptron.__set_test_and_train_sets(self._dataset.drop(self.target_name, axis = 1), self._dataset[self.target_name], self._test_size)     
        self.train_set.insert(0,'Bias',1)
        self.test_set.insert(0,'Bias',1)

    @staticmethod
    def __set_test_and_train_sets(data, target, test_size):
        """Split data into train and test sets"""
        X_train, X_test, y_train, y_test = train_test_split(data, target, test_size = test_size , random_state = 0 )
        return X_train, X_test, y_train, y_test        

    @staticmethod
    def __activation_output(treshold, weights, row):
        
        a = sum([x * y for x,y in zip(weights,row)])
        return 1 if (a > treshold) else 0  


    def find_weights(self):
        RMSEprev = 100
        cols = self.train_set.shape[1]
        weights = np.random.rand(cols)
        while True:
            globalErr = 0.0
            for ind in self.train_set.index:
                # compute activation output
                row = self.train_set.loc[ind]
                output = Perceptron.__activation_output(self.treshold,weights,row)
                error = self.train_target.loc[ind] - output
                weights = weights + error * self.learning_rate  * row
                globalErr += (error ** 2)

            RMSE = np.sqrt(globalErr) / self.train_set.shape[0]
            
            if (RMSEprev - RMSE) > 0.0001:
                break
            else:
                RMSEprev = RMSE
        return weights

    def result(self, weights):
        results = []
        for ind in self.test_set.index:
            row = self.test_set.loc[ind]
            result = Perceptron.__activation_output(self.treshold,weights,row)
            results.append(result)
        return pd.Series(results)
    
    def accurancy(self, result):
        fitting = np.where(self.test_target.values == result,1,0 )
        return fitting.sum()/ float(len(result))    

