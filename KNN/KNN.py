import pandas as pd
import numpy as np


class KNN():
    """This class represents the K Nearest Neighbourhood
    I implemented this class to get a better understanding of KNN.
    The algorithm is good for classification problem with two classes. 
    * k = number of nearest neighbourhoods, the best choise would be odd number
    * df_file = (str) path to the data, for instance, in csv format
    * target_name = (str) name of column with the target
    * test_range = (int) numerator, how to split data for test and train set.
                    For example, if test_range = 3, then test_range is 1/3 of all the data.
    * sep = (str) how to split data in file; default None
    * header = (int) no. of row with names of columns; default " "
    * index_col = (int) no. of column with index values; default None
    * columns = (list) if there is no header in data, set the list with colomns names 


    """


    def __init__(self, k, df_file, target_name, test_range, sep = " ", header = None, index_col = None, columns = None):
        self._test_range = test_range
        self.k = k
        self.target_name = target_name

        self._data = pd.read_csv(df_file, sep = sep, header = header, index_col = index_col)
        if columns is not None:
            self._data.columns = columns
        self.train_set, self.test_set = KNN.__set_test_and_train_sets(self._data, self._test_range)
       

    @staticmethod
    def __set_test_and_train_sets(data, test_range):
        rows = data.shape[0] 
        range_all = list(range(rows))
        range_test = list(range(0,rows,test_range))
        range_train = [x for x in range_all if x not in range_test]
    
        test_set = data.iloc[range_test]
        test_set.reset_index(drop = True, inplace = True)
        train_set = data.iloc[range_train]
        train_set.reset_index(drop = True, inplace = True)
        return train_set, test_set    


    def result_test_set(self):
        k = self.k
        target_name = self.target_name
        trainset = self.train_set
        testset = self.test_set
        result = []
        testset_attr = testset.drop(target_name, axis = 1)
        for row in testset.index:
            testset_row = testset_attr.iloc[row]
            distance = KNN.__euklidean_dist(trainset, testset_row, target_name)
            closest = distance.nsmallest(k).index
            train_closest = trainset.iloc[closest][target_name]
            counts = train_closest.value_counts()
            result_row = counts.idxmax()
            result.append(result_row)
        total_result = pd.Series(result, name = 'Result')
        return total_result 


    @staticmethod
    def __euklidean_dist(trainset, testrow , targetname):
        trainset_attr = trainset.drop(targetname, axis = 1)
        a = trainset_attr - testrow
        b = (a)**2
        c = np.sum(b, axis = 1 )
        result = np.power(c, 0.5)
        return result

   
    def final_fitting(self, result_test_set):
        test_set = self.test_set
        target_name = self.target_name
        result = pd.concat([test_set[target_name], result_test_set] , axis = 1)
        result["Compare"] = np.where(result.ix[:,0] == result.ix[:,1], 1, 0)
        final_fitting = result["Compare"].sum()/ float(result.shape[0] )
        return final_fitting


