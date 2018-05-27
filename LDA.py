import pandas as pd
import numpy as np
from numpy.linalg import inv
from sklearn.model_selection import train_test_split

class LDA():
    """This class represents the Linear Discriminant Analysis .
    I implemented this class to get a better understanding of LDA.
    The algorithm is good for classification problem with two classes.

    * target_name = (str) name of column with the target
    * test_size = (float/int/None) If float, should be between 0.0 and 1.0,
                    and represent the proportion of the dataset to include in the test split. 
                    If int, represents the absolute number of test samples. 
                    If None, the value is set to the complement of the train size. 
                    By default, the value is set to 0.25. 
    * file_path = (str) path to the daa, for instance, in csv format
    * df_sep = (str) how to split data in file; default: " "
    * df_header = (int) no. of row with names of columns; default None
    * index_col = (int) no. of column with index values; default None
    * columns = (list) if there is no header in data, set the list with colomns names; default None 


    """
    def __init__(self, target_name, test_size, file_path, sep = " ", header = None, index_col = None, columns = None):
        self._test_size = test_size
        self.target_name = target_name

        self._dataset = pd.read_csv(file_path, sep = sep, header = header)
        if columns is not None:
            self._dataset.columns = columns
        # self._col_target = self.train_target
        self.train_set,self.test_set, self.train_target, self.test_target = LDA.__set_test_and_train_sets(self._dataset.drop(self.target_name, axis = 1), self._dataset[self.target_name], self._test_size)     
        self._target_unique = self._dataset[self.target_name].unique()

    @staticmethod
    def __set_test_and_train_sets(data, target, test_size):
        """Split data into train and test sets"""
        X_train, X_test, y_train, y_test = train_test_split(data, target, test_size = test_size , random_state = 0 )
        return X_train, X_test, y_train, y_test

    def count_and_prob(self):
        '''DataFrame with counted number of target and probability of appeared target'''
        df_count_and_prob = pd.DataFrame( index = self._target_unique, columns = ['Count', 'Probability' ])
        df_count_and_prob['Count'] = self.train_set.groupby(self.train_target).count()
        df_count_and_prob['Probability'] = df_count_and_prob['Count'] / df_count_and_prob['Count'].sum()
        return df_count_and_prob

    def mean_vectors(self):
        """Create mean values for each attribute grouped be target"""
        return self.train_set.groupby(self.train_target).mean()

    def covariance_matrix(self):
        """Create covariance matrix for atributes and targets"""
        return self.train_set.groupby(self.train_target).cov()

    def pooled_cov_matrix(self, df_count_and_prob, covs):
        """Create pooled covariance matrix"""
        part = float(1)/ df_count_and_prob['Count'].sum()
        pooled_cov_part = 0
        for ind in df_count_and_prob.index:
            pooled_cov_part += covs.loc[ind] * df_count_and_prob.loc[ind,'Count']
        pooled_cov = (part * pooled_cov_part).values    
        return pooled_cov


    def beta(self, mean_vectors, pooled_cov):
        """Calculate beta coefficients"""
        x = (mean_vectors.loc[self._target_unique[0]] - mean_vectors.loc[self._target_unique[1]]).transpose().values
        return inv(pooled_cov).dot(x)

    def Mahalanobis(self,mean_vectors, beta):
        """Calculate Mahalanobis coefficient"""
        x = (mean_vectors.loc[self._target_unique[0]] - mean_vectors.loc[self._target_unique[1]]).transpose().values
        return np.sqrt(beta.T.dot(x))       


    def check_test_set(self, count_and_prob, mean_vectors, beta):
        prob = np.log(count_and_prob.loc[self._target_unique[0]]['Probability'] /count_and_prob.loc[self._target_unique[1]]['Probability']  )
        mean2 = ((mean_vectors.loc[self._target_unique[0]] + mean_vectors.loc[self._target_unique[1]])/2).values
        result = []
        for row in self.test_set.index:
            x1 = self.test_set.loc[row].values 
            if beta.T.dot(x1 - mean2) > prob:
                result.append(self._target_unique[0])
            else:
                result.append(self._target_unique[1])
        results = pd.Series(result, name = "Result")
        return results

    def result_fitting(self, result_test_set):
        self.test_target.reset_index(drop = True, inplace = True)
        result = pd.concat([self.test_target, result_test_set] , axis = 1)
        result["Compare"] = np.where(result.ix[:,0] == result.ix[:,1], 1, 0)
        return result["Compare"].sum()/ float(result.shape[0])

  