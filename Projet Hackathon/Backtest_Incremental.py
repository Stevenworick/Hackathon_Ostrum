#!/usr/bin/env python
# coding: utf-8

# In[31]:


import pandas as pd 
import numpy as np

from  statsmodels.tsa.stattools import acf 
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression,LogisticRegressionCV,RidgeClassifier,RidgeClassifierCV
from sklearn.ensemble  import AdaBoostClassifier, GradientBoostingClassifier,RandomForestClassifier,BaggingClassifier 
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import BernoulliNB,CategoricalNB
from xgboost import XGBClassifier
import time
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV

from sklearn.linear_model import ElasticNet
from lightgbm import LGBMClassifier
from ml4pmt.plot import line, bar, heatmap

from statsmodels.tsa.tsatools import lagmat
import seaborn as sns
import warnings
from tqdm import tqdm



# ### 1) Strategies



# In[43]:


def strategy_long_only(Capital, Qty, Vix, Pred):
    '''
    capital : The cash available in the portfolio
    Qty : The Vix's quantity in the portfolio
    Vix : Vix's price 
    Pred : The prediction given ( if 1 buy vix, if 0 and there is Vix in the portfolio, we sell it)
    '''
    
    if Qty != 0 and Pred == 0:
        Capital = Qty * Vix
        Qty = 0
    if Qty == 0 and Pred == 1:
        Qty = Capital / Vix
        Capital = 0
        
    Portfolio_value = Capital + Qty * Vix
    return Capital,Qty,Portfolio_value

def strat_long_short(Capital, Qty, Vix, Proba, Short):
    '''
    capital : The cash available in the portfolio
    Qty : The Vix's quantity in the portfolio
    Vix : Vix's price 
    Proba : The prediction given ( if proba >0.55 buy vix, if <0.45 short vix)
    Short : Tell if we have a short position in our portfolio
    '''

    if Proba>0.55 and Capital!= 0 :
        #if we have a short position, first we close it
        if Short == True:
            #We were short so we have to buy back vix
            Capital += Qty * Vix
        #Then we buy vix
        Qty = Capital/Vix
        Capital = 0
        Short = False
    
    if Proba< 0.45 and Short== False:
        #If we are long vix, first we close our position
        if Qty !=0:
            Capital += Qty*Vix

        Qty = - Capital / Vix ##### Qty negative calcul portfolio = Capital + qty * vix
        Capital += - Qty*Vix
        Short = True
        
    Portfolio_value = Capital + Qty * Vix
    return Capital,Qty, Short, Portfolio_value

def strategy_long_only_SP500(Capital, Qty, SP500, Proba):
    '''
    capital : The cash available in the portfolio
    Qty : The Vix's quantity in the portfolio
    Vix : Vix's price 
    Proba : The prediction given ( if <0.45 buy SP500, if >0.55 and there is SP500 in the portfolio, we sell it)
    '''
    if Qty != 0 and Proba > 0.55:
        Capital = Qty * SP500
        Qty = 0
    if Qty == 0 and Proba < 0.45:
        Qty = Capital / SP500
        Capital = 0
    
    Portfolio_value = Capital + Qty * SP500
    return Capital,Qty, Portfolio_value


# ### 2) Data Initialisation 

# In[44]:


class Data_Initialisation:
    def __init__(self,
                 end_train_date = '2004-12-31',
                 begin_test_date = '2005-01-31',
                hurst = False):
        if hurst :
            data = pd.read_csv(r"df_merged_data_hurst.csv",sep=';',decimal=',')
            data = data.set_index("Date")
            data.index = pd.to_datetime(data.index, format = '%d/%m/%Y').strftime('%Y-%m-%d')
        
        else :
            data = pd.read_csv(r"df_merged_data.csv")
            data = data.set_index("Date")
            
        self.data = data
        self.end_train_date = end_train_date
        self.begin_test_date = begin_test_date
        
    def preprocessed_time_serie_split(self, lag = False) :
        '''
        Here we process the data in the same way as in the article
        If Lag = True, datas will be lagged
        '''
        
        self.data['LEVERAGE'] = (self.data['LEVERAGE'] - self.data['LEVERAGE'].mean()) / self.data['LEVERAGE'].std() 
        self.data['DOLLAR'] = self.data['DOLLAR'].pct_change()
        self.data['OIL'] = self.data['OIL'].pct_change()
        self.data =  self.data.tail(self.data.shape[0]-1)
        self.data["VIX_t_1"] = self.data["VIX"].shift(-1)
        self.data["Y_pred"] = 1*(self.data["VIX"] < self.data["VIX_t_1"]) 
        del self.data["VIX_t_1"]
        
        data_train = self.data[:self.end_train_date]
        data_test = self.data[self.begin_test_date:]
        
        #Here we lag the datas
        
        self.X_train = data_train.loc[:, ~data_train.columns.isin(['Y_pred'])]
        if lag:
            self.X_train = lagmat(self.X_train,maxlag=3, use_pandas= True)
        self.y_train = data_train["Y_pred"]
        self.X_test = data_test.loc[:, ~data_test.columns.isin(['Y_pred'])]
        Vix = self.X_test["VIX"]
        Sp500 = self.X_test["SP500"]
        if lag:
            self.X_test = lagmat(self.X_test,maxlag=3, use_pandas= True)
        self.y_test =data_test["Y_pred"]
        
        return self.X_train, self.X_test, self.y_train, self.y_test, Vix, Sp500

        


# In[73]:


class Portfolio:
    def __init__(self,
                Initial_Cash,
                 Initial_Date
                ):
        self.cash = Initial_Cash
        self.asset_qty = 0
        self.portfolio_value = {Initial_Date : Initial_Cash}
        self.short_position = False
    
    def update_portfolio_value(self,
                               cash,
                             Date,
                            portfolio_value,
                               asset_qty,
                               short_position = False
                            ):
        self.cash = cash
        self.portfolio_value[Date] = portfolio_value
        self.asset_qty = asset_qty
        self.short_position = short_position
        
    def returns_portfolio(self
                         ):
        returns_df = pd.DataFrame.from_dict(self.portfolio_value, orient = 'index')
        returns_df = returns_df.pct_change()
        #returns_df =  returns_df.tail(returns_df[0]-1)
        
        return returns_df
            
        


# In[80]:


class Backtester:
    def __init__(
        self,
        estimator,
        grids,
        portfolios
    ):

        self.estimator = estimator
        self.grids = grids
        self.portfolio_1 = portfolios[0]
        self.portfolio_2 = portfolios[1]
        self.portfolio_3 = portfolios[2]

    
    def incremental_learning(self, features_train, target_train, features_test, target_test, Vix, Sp500, alpha = 0.75, verbose = False):
        self.features_selected = pd.Series(0, index = features_train.columns)
        warnings.filterwarnings("ignore")
        for i in tqdm(range(len(features_test))):

            start_time=time.time()
            # Elastic Net 

            regr = ElasticNet(random_state=0,alpha = 0.75)

            regr.fit(features_train, target_train)

            coef = pd.Series(regr.coef_, index = features_train.columns)
            coef_ = coef[coef!=0].to_frame().index.value_counts()
            self.features_selected =pd.concat([self.features_selected, coef_], axis=1).fillna(0).sum(axis=1) 
            features_train_filtred = features_train[coef[coef!=0].index.to_list()] # We keep only non zero coefficients

            for name, model in self.estimator:

                if "AUC" in name : 
                    clf = GridSearchCV(model, self.grids[name], scoring='roc_auc', cv=3, n_jobs=-1)
                elif "Class" in name : 
                    clf = GridSearchCV(model, self.grids[name], scoring='CLASS', cv=3, n_jobs=-1)
                elif "Dev" in name : 
                    clf = GridSearchCV(model, self.grids[name], scoring='DEV', cv=3, n_jobs=-1)
                else:    
                    clf = GridSearchCV(model, self.grids[name], scoring="accuracy", cv=3, n_jobs=-1)

                clf.fit(features_train_filtred, target_train)
                
                if verbose:
                    print("Elastic Net picked " + str(sum(coef != 0)) + " variables and eliminated the other " +  str(sum(coef == 0)) + " variables")
                    print('\033[1m' + " Results for {}:".format(name) + '\033[0m')
                    print(' Returned hyperparameter: {}'.format(clf.best_params_))
                    print(' Best classification score in validation set is: {}'.format(round(clf.best_score_,3)*100))
                    print("--------- %s secondes ---------" % (round(time.time() - start_time,2)))
                    print('-----------------------------------------------------')



                features_test_filtred = features_test[coef[coef!=0].index.to_list()]
                # Prédiction sur une seule observation
                pred  = clf.predict(pd.DataFrame(features_test_filtred.iloc[i,:]).T)
                proba = clf.predict_proba(pd.DataFrame(features_test_filtred.iloc[i,:]).T)[0][1] 
                date = features_test_filtred.index[i]

            # Stratégie

                Cash1,Qty1,Pfolio_1 = strategy_long_only(self.portfolio_1.cash, self.portfolio_1.asset_qty, Vix.loc[date], pred)
                Cash2,Qty2,Short,Pfolio_2 = strat_long_short(self.portfolio_2.cash, self.portfolio_2.asset_qty, Vix.loc[date], proba, self.portfolio_2.short_position)
                Cash3,Qty3,Pfolio_3 = strategy_long_only_SP500(self.portfolio_3.cash, self.portfolio_3.asset_qty,  Sp500.loc[date], proba)
    
                self.portfolio_1.update_portfolio_value(Cash1, date, Pfolio_1, Qty1)
                self.portfolio_2.update_portfolio_value(Cash2, date, Pfolio_2, Qty2, Short)
                self.portfolio_3.update_portfolio_value(Cash3, date, Pfolio_3, Qty3)

                # Append une observation au Train 
                features_train.loc[date] = features_test.iloc[i,]
                target_train.loc[date] = target_test.iloc[i]
                
    def plot_pnl(self,
              sp500,
              VIX):
        '''
        Compute and plot pnl for each strategy
        '''
        pfolio_1_returns = self.portfolio_1.returns_portfolio()
        pfolio_2_returns = self.portfolio_2.returns_portfolio()
        pfolio_3_returns = self.portfolio_3.returns_portfolio()
        
        sp500_return= sp500.pct_change()
        vix_return = VIX.pct_change()
        pnls={'strat 1 :' : pfolio_1_returns, 'strat 2 :' : pfolio_2_returns, 'strat 3 :' : pfolio_3_returns, 'sp500' : sp500_return, 'VIX' : vix_return}
        line(pnls, cumsum=True, start_date='2005', title='Cumulative pnl for different look-back windows (in month)', year_only = True, figsize = (16,9))
        
    def heatmap_features(self):
                
        features = self.features_selected.to_frame()
        features.rename({'0': 'Number of selection by elatic net'}, axis=1, inplace=True)
        features = features.loc[~(features==0).all(axis=1)]
        features.sort_index(level=1, ascending=False, inplace=True)
        
        ax = plt.axes()
        sns.heatmap(features, annot=True, fmt="g", ax = ax)
        ax.set_title('Number of selection by elatic net')
        plt.show()
        
        

