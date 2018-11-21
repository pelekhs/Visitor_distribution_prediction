# -*- coding: utf-8 -*-
"""
Created on Wed Jun 13 12:48:44 2018

@author: spele
"""
import matplotlib.pyplot as plt
from math import sqrt
from sklearn.metrics import r2_score
import numpy as np

class model_evaluation_and_plots():
    def __init__(self,X_train_sc_flat,X_test_sc_flat,y_train_sc,ytrue, y_train,
                y_test, sc_cl):
        self.X_train_sc_flat = X_train_sc_flat
        self.X_test_sc_flat = X_test_sc_flat
        self.y_train_sc=y_train_sc
        self.ytrue=ytrue
        self.y_train=y_train
        self.y_test=y_test
        self.sc_cl=sc_cl

    def evaluate(self,title, ytest_hat_sc, ytrain_hat_sc, yhat_sc):
        #inverse transform yhat and compare it with y_test original(not scaled)
        ytest_hat = self.sc_cl.inverse_transform(ytest_hat_sc).reshape((len(ytest_hat_sc))).round(0)
        ytrain_hat = self.sc_cl.inverse_transform(ytrain_hat_sc).round(0).reshape((len(ytrain_hat_sc)))                      
        yhat = self.sc_cl.inverse_transform(yhat_sc).round(0).reshape((len(yhat_sc)))
    
        ##ERRORS
        #standard error of regression
        Sete = sqrt((1/(len(self.y_test)-2))*sum((ytest_hat[:]-self.y_test[:])**2))
        print("Standard Error of Regression(test) = ",Sete)
        Setr = sqrt((1/(len(self.y_train)-2))*sum((ytrain_hat[:]-self.y_train[:])**2))
        print("Standard Error of Regression(train) = ",Setr)
        Se = sqrt((1/(len(self.ytrue)-2))*sum((yhat[:]-self.ytrue[:])**2))
        print("Standard Error of Regression(all) = ",Se)
        
    #    #goodness of fit of regression (is between (0,1) and we want it to be close to 1)
    #    gof = sum((ytest_hat[:]-np.mean(y_test))**2)/sum((y_test[:]-np.mean(y_test))**2)
    #    print("Goodness of Fit (test) = ",gof)
    #    gof = sum((ytrain_hat[:]-np.mean(y_train))**2)/sum((y_train[:]-np.mean(y_train))**2)
    #    print("Goodness of Fit (train) = ",gof)
    #    gof = sum((yhat[:]-np.mean(ytrue))**2)/sum((ytrue[:]-np.mean(ytrue))**2)
    #    print("Goodness of Fit (all) = ",gof)
      
        #Rsquared
        R2te = r2_score(self.y_test, ytest_hat, sample_weight=None, multioutput='uniform_average')
        print("r2(test) = ", R2te)
        R2tr = r2_score(self.y_train, ytrain_hat, sample_weight=None, multioutput='uniform_average')
        print("r2(train) = ", R2tr)
        R2 = r2_score(self.ytrue, yhat, sample_weight=None, multioutput='uniform_average')
        print("r2(all) = ", R2)
        
        #mape

        def mean_absolute_percentage_error(y_true, y_pred): 
            k=np.where(y_true!=0)
            y_true, y_pred = np.array(y_true), np.array(y_pred)
            return np.mean(np.abs((y_true[k] - y_pred[k]) / y_true[k])) * 100
   
        mapete = mean_absolute_percentage_error(self.y_test, ytest_hat)
        print("mape(test) = ", mapete)
        mapetr = mean_absolute_percentage_error(self.y_train, ytrain_hat)
        print("mape(train) = ", mapetr)
        mape = mean_absolute_percentage_error(self.ytrue, yhat)
        print("mape(all) = ", mape,"\n")
       
        #after evaluation eliminate negative predictions. There is no meaning in them
        ytest_hat[ytest_hat < 0] = 0
        ytrain_hat[ytrain_hat < 0] = 0
        yhat[yhat < 0] = 0
        
        #plot results
        plt.figure(title)
        plt.plot(range(len(self.ytrue)),self.ytrue,color = 'black', label = "True")
        plt.plot(range(len(self.ytrue)),yhat, color = 'red',label = "Hat")
        plt.legend()
        
        return yhat, ytrain_hat, ytest_hat, Sete, Setr,Se, R2te, R2tr, R2,\
                mapete, mapetr, mape
          
        
    ################################       SVR      #################################
    def SVR(self,svr_kernel):
        # Fitting different SVR kernels to the dataset 
        print("SVR kernel = " + svr_kernel)
        from sklearn.svm import SVR
        regressor = SVR(kernel = svr_kernel)
        regressor.fit(self.X_train_sc_flat, self.y_train_sc)
        #predict
        ytest_hat_sc = regressor.predict(self.X_test_sc_flat).reshape(-1,1)   
        ytrain_hat_sc = regressor.predict(self.X_train_sc_flat).reshape(-1,1)
        yhat_sc = regressor.predict(np.concatenate([self.X_train_sc_flat,self.X_test_sc_flat])).reshape(-1,1)   
        #evaluate and plot
        return self.evaluate("SVR: " + svr_kernel,
                ytest_hat_sc, ytrain_hat_sc, yhat_sc)
    ###############################  RANDOM FOREST  ##################################
    
    def RF(self,estimator=1, max_depth=None, min_samples_split=2):
        from sklearn import ensemble
        print("Random Forest estimators = " + str(estimator))
        regressor = ensemble.RandomForestRegressor(n_estimators = estimator,max_depth=max_depth, min_samples_split=min_samples_split)
        regressor.fit(self.X_train_sc_flat, self.y_train_sc[:len(self.X_train_sc_flat)])
        #predict
        ytest_hat_sc = regressor.predict(self.X_test_sc_flat).reshape(-1,1)   
        ytrain_hat_sc = regressor.predict(self.X_train_sc_flat).reshape(-1,1)
        yhat_sc = regressor.predict(np.concatenate([self.X_train_sc_flat,self.X_test_sc_flat])).reshape(-1,1)
        #evaluate and plot
        return self.evaluate("Random Forest with estimators: "+ str(estimator),
                 ytest_hat_sc, ytrain_hat_sc, yhat_sc)
        
    ###############################    POLYNOMIAL    ################################
    def poly(self,deg):
        from sklearn.linear_model import LinearRegression
        from sklearn.preprocessing import PolynomialFeatures
        print("Polynomial Regression with degree = " + str(deg))
        poly_reg = PolynomialFeatures(degree = deg)
        X_poly = poly_reg.fit_transform(self.X_train_sc_flat)
        poly_reg.fit(X_poly, self.y_train_sc[:len(self.X_train_sc_flat)])
        regressor = LinearRegression()
        regressor.fit(X_poly, self.y_train_sc[:len(self.X_train_sc_flat)])
        
        #predict
        X_poly_test = poly_reg.fit_transform(self.X_test_sc_flat)
        ytest_hat_sc = regressor.predict(X_poly_test).reshape(-1,1)   
        ytrain_hat_sc = regressor.predict(X_poly).reshape(-1,1)
        yhat_sc = regressor.predict(np.concatenate([X_poly,X_poly_test])).reshape(-1,1)   
        
        #evaluate and plot
        return self.evaluate("Polynomial regression with degree:"+ str(deg),ytest_hat_sc,
                 ytrain_hat_sc, yhat_sc)
    #############################     MLP    ###########################################
    def MLP(self):
        from sklearn.neural_network import MLPRegressor
        print("Multi Layer Perceptron")
        mlp = MLPRegressor()
        mlp.fit(self.X_train_sc_flat,self.y_train_sc)
        
        #predict
        ytest_hat_sc = mlp.predict(self.X_test_sc_flat).reshape(-1,1)   
        ytrain_hat_sc = mlp.predict(self.X_train_sc_flat).reshape(-1,1)
        yhat_sc = mlp.predict(np.concatenate([self.X_train_sc_flat,self.X_test_sc_flat])).reshape(-1,1)   
        
        #evaluate and plot
        return self.evaluate("Multilayer perceptron",ytest_hat_sc, ytrain_hat_sc, yhat_sc)
    ################################     KLAMA     ####################################
    def shift(self,timestep,output_cols,indexes,testing_day,years='all'):
        print("Shift output")
        mean = []
        for i in range(timestep,len(output_cols)):
            tk = 0
            for k in range(timestep):
                tk = tk+ output_cols[i-k-1]
            tk = tk/timestep
            mean.append(tk)
        if years!='all':
            yhat = np.asarray(mean)#yhat
            ytrue = output_cols[timestep:]
            third = (np.argwhere(indexes == np.amax(indexes))+1).reshape(3)
            if testing_day==3:
                y_test = ytrue[third[1]:]
                ytest_hat = yhat[third[1]:]
                y_train = ytrue[:third[1]]
                ytrain_hat = yhat[:third[1]]
            elif testing_day==2:
                y_test = ytrue[third[0]:third[1]]
                ytest_hat = yhat[third[0]:third[1]]
                y_train = np.concatenate([ytrue[:third[0]],ytrue[third[1]:]])
                ytrain_hat = np.concatenate([yhat[:third[0]],yhat[third[1]:]])
            else:
                y_test = ytrue[:third[0]]
                ytest_hat = yhat[:third[0]]
                y_train = ytrue[third[0]:]
                ytrain_hat = yhat[third[0]:]
        else:
            yhat = np.asarray(mean)#yhat
            ytrue = output_cols[timestep:]
            sixth = (np.argwhere(indexes == np.amax(indexes))+1).reshape(6)
            if testing_day==6:
                y_test = ytrue[sixth[4]:]
                ytest_hat = yhat[sixth[4]:]
                y_train = ytrue[:sixth[4]]
                ytrain_hat = yhat[:sixth[4]]
            elif testing_day==5:
                y_test = ytrue[sixth[3]:sixth[4]]
                ytest_hat = yhat[sixth[3]:sixth[4]]
                y_train = np.concatenate([ytrue[:sixth[3]],ytrue[sixth[4]:]])
                ytrain_hat = np.concatenate([yhat[:sixth[3]],yhat[sixth[4]:]])
            elif testing_day==4:
                y_test = ytrue[sixth[2]:sixth[3]]
                ytest_hat = yhat[sixth[2]:sixth[3]]
                y_train = np.concatenate([ytrue[:sixth[2]],ytrue[sixth[3]:]])
                ytrain_hat = np.concatenate([yhat[:sixth[2]],yhat[sixth[3]:]])
            elif testing_day==3:
                y_test = ytrue[sixth[1]:sixth[2]]
                ytest_hat = yhat[sixth[1]:sixth[2]]
                y_train = np.concatenate([ytrue[:sixth[1]],ytrue[sixth[2]:]])
                ytrain_hat = np.concatenate([yhat[:sixth[1]],yhat[sixth[2]:]])
            elif testing_day==2:
                y_test = ytrue[sixth[0]:sixth[1]]
                ytest_hat = yhat[sixth[0]:sixth[1]]
                y_train = np.concatenate([ytrue[:sixth[0]],ytrue[sixth[1]:]])
                ytrain_hat = np.concatenate([yhat[:sixth[0]],yhat[sixth[1]:]])
            else:
                y_test = ytrue[:sixth[0]]
                ytest_hat = yhat[:sixth[0]]
                y_train = ytrue[sixth[0]:]
                ytrain_hat = yhat[sixth[0]:]
         
        ##ERRORS
        #standard error of regression
        Sete = sqrt((1/(len(y_test)-2))*sum((ytest_hat[:]-y_test[:])**2))
        print("Standard Error of Regression(test) = ",Sete)
        Setr = sqrt((1/(len(y_train)-2))*sum((ytrain_hat[:]-y_train[:])**2))
        print("Standard Error of Regression(train) = ",Setr)
        Se = sqrt((1/(len(ytrue)-2))*sum((yhat[:]-ytrue[:])**2))
        print("Standard Error of Regression(all) = ",Se)
        
        #Rsquared
        R2te = r2_score(y_test, ytest_hat, sample_weight=None, multioutput='uniform_average')
        print("r2(test) = ", R2te)
        R2tr = r2_score(y_train, ytrain_hat, sample_weight=None, multioutput='uniform_average')
        print("r2(train) = ", R2tr)
        R2 = r2_score(ytrue, yhat, sample_weight=None, multioutput='uniform_average')
        print("r2(all) = ", R2)
        
        #mape
        def mean_absolute_percentage_error(y_true, y_pred): 
            k=np.where(y_true!=0)
            y_true, y_pred = np.array(y_true), np.array(y_pred)
            return np.mean(np.abs((y_true[k] - y_pred[k]) / y_true[k])) * 100
   
        mapete = mean_absolute_percentage_error(y_test, ytest_hat)
        print("mape(test) = ", mapete)
        mapetr = mean_absolute_percentage_error(y_train, ytrain_hat)
        print("mape(train) = ", mapetr)
        mape = mean_absolute_percentage_error(ytrue, yhat)
        print("mape(all) = ", mape,"\n")
        
        #plot results
        plt.figure("Shifted by 1 output to input")
        plt.plot(range(len(ytrue)),np.concatenate([y_train,y_test]),color = 'black', label = "True")
        plt.plot(range(len(ytrue)),np.concatenate([ytrain_hat,ytest_hat]), color = 'red',label = "Hat")
        plt.legend()
        return yhat, ytrain_hat, ytest_hat, Sete, Setr,Se, R2te, R2tr, R2,\
                mapete, mapetr, mape