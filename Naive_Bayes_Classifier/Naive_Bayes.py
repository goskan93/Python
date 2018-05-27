import numpy as np
import pandas as pd


class Naive_Bayes_Classifier():
    """This class represents the Naive Bayes Classifier Algorithm.
    I implemented this class to get a better understanding of NBC.
    The algorithm is good for classification problem with categorital data. 
    * df_file = (str) path to the data, for instance, in csv format
    * target_name = (str) name of column with the target
    * test_range = (int) numerator, how to split data for test and train set.
                    For example, if test_range = 3, then test_range is 1/3 of all the data.
    * df_sep = (str) how to split data in file; default " "
    * df_header = (int) no. of row with names of columns; default None
    * index_col = (int) no. of column with index values; default None
    * columns = (list) if there is no header in data, set the list with colomns names 


    """

    def __init__(self, df_file, target_name, test_range, df_sep = ",", df_header = None, index_col=None, columns = None):
        """Create class of Naive Bayes Classifier"""
        self.target_name = target_name
        self._test_range = test_range
        self._data = pd.read_csv(df_file, df_sep, df_header, index_col)
        if columns is not None:
            self._data.columns = columns
        
        self.train_set, self.test_set = Naive_Bayes_Classifier.__set_test_and_train_sets(self._data, self._test_range)

    @staticmethod
    def __set_test_and_train_sets(data, test_range):
        """Split data into train and test sets"""
        rows = data.shape[0] 
        range_all = list(range(rows))
        range_test = list(range(0,rows,test_range))
        range_train = [x for x in range_all if x not in range_test]
    
        test_set = data.iloc[range_test]
        test_set.reset_index(drop = True, inplace = True)
        train_set = data.iloc[range_train]
        train_set.reset_index(drop = True, inplace = True)
        return train_set, test_set

        
    def set_freq_table(self, train_data, target_name):
        """Building frequency tables (confusion matrix) for each column and target.
         Information are set in dictionary.
         

         """
        freq_tables = {}
        for col in train_data.columns:
            if col != target_name:
                crosstab =  pd.crosstab(train_data[col], train_data[target_name]) 
                if crosstab.isin([0]).any().any():
                    crosstab += 1
                crosstab["All"] = crosstab.sum(axis = 1)   
                crosstab.loc["All"] =  crosstab.sum() 
                freq_tables[col] = crosstab
        return freq_tables 


    @staticmethod
    def __likelihood_tab(crosstab):
        """Set likelihood table for single column and target"""
        likelihood_t = pd.DataFrame( columns = crosstab.columns, index = crosstab.index)
        for col_name in crosstab.columns:
            if col_name == "All":
                for row_name in crosstab.index:
                    likelihood_t.loc[row_name,col_name] = float(crosstab.loc[row_name,col_name]) / crosstab.loc["All",col_name]
            else:
                for row_name in crosstab.index:
                    if row_name == "All":
                        likelihood_t.loc[row_name,col_name] = float(crosstab.loc[row_name,col_name]) / crosstab.loc[row_name, "All"]
                    else:
                        likelihood_t.loc[row_name,col_name] = float(crosstab.loc[row_name,col_name]) / crosstab.loc["All",col_name]
    
        return likelihood_t



    def set_likelihood_table(self,freq_table):
        """Set likelihood tables based on frequency tables for each column and target.
        Information are set in dictionary.


        """
        likelihood_tab = {}
        for key, crosstab in freq_table.items():
            likelihood_t = Naive_Bayes_Classifier.__likelihood_tab( crosstab)
            likelihood_tab[key] = likelihood_t 
        return likelihood_tab   


    @staticmethod
    def __normalize(results):
        total = sum(results.values())
        normalized = {k: v / total for k, v in results.items()}
        return normalized

    @staticmethod   
    def __best_choice(norm_dict):
        import operator
        return max(norm_dict.items(), key = operator.itemgetter(1))[0]

    
    def test_likelihood(self, test_data, likelihood_tab, target_name):
        """Count the Bayesian probability for test data"""
        result_list = []
        targets = test_data[target_name].unique()
        for row in test_data.index:
            result_row = {}
            for tar in targets:
                likelihood_of_tar = 1
                for col in test_data.columns:
                    if col != target_name:
                        freq_tab = likelihood_tab.get(col)
                        var = test_data.loc[row,col]
                        posterior_prob = freq_tab.loc[var,tar] * freq_tab.loc["All",tar] / freq_tab.loc[var,"All"]
                        likelihood_of_tar *= posterior_prob
                likelihood_of_tar *=  freq_tab.loc["All",tar]
                result_row[tar] = likelihood_of_tar
            result_row = Naive_Bayes_Classifier.__best_choice(Naive_Bayes_Classifier.__normalize(result_row))
            result_list.append(result_row)
        results = pd.Series(result_list, name = "Result")
        return results

    def final_fitting(self, test_data, target_name, col_result):
        result = pd.concat([test_data[target_name] ,col_result ] , axis=1)
        result["Compare"] = np.where(result.ix[:,0] == result.ix[:,1], 1, 0)
        final_fitting = result["Compare"].sum()/ float(result.shape[0] ) 
        return final_fitting


