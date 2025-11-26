
from abc import ABC, abstractmethod
import os, os.path
import numpy as np 
import pandas as pd
import requests , zipfile
import subprocess

import sys
import numpy as np
from scipy.optimize import minimize_scalar
from sklearn.decomposition import PCA
import statsmodels.api as sm
import random

import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Layer
from tensorflow.keras.metrics import binary_accuracy
from tensorflow.keras.losses import binary_crossentropy as keras_binary_crossentropy

import numpy as np
from tensorflow.keras import Input
from tensorflow.keras import Model as keras_Model
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, TerminateOnNaN
from tensorflow.keras.layers import Dense, Concatenate, Dropout
from tensorflow.keras.optimizers import Adam
try:
    from tensorflow.keras.optimizers.legacy import SGD
except ImportError:
    from tensorflow.keras.optimizers import SGD
from tensorflow.keras.regularizers import l2
from tensorflow.keras.layers import Layer

from keras import regularizers

import pandas as pd

# entropy balance
class ebal_bin:
    """
    Implementation of Entropy Balancing for binary treatment
    
    Author: Eddie Yang, based on work of Hainmueller (2012) and Xu & Yang (2021)
    
    Params:
    coefs: Lagrangian multipliers, refer to Hainmueller (2012)
    max_iterations: maximum number of iterations to find the solution weights, default 500
    constraint_tolerance: tolerance level for covariate difference between the treatment and control group, default 1e-4
    print_level: level of details to print out
    lr: step size, default 1. Increase to make the algorithm converge faster (at the risk of exploding gradient)

    Output:
    converged: boolean, whether the algorithm converged
    maxdiff: maximum covariate difference between treatment and control groups
    w: solution weights for the control units
    wls: output from weighted OLS regression

    Current version: 1.0.2
    updates:
    1. add PCA and WLS options
    2. fix wrong initial coefs 
    3. add weights for treated units
    """

    def __init__(self, 
        coefs = None, 
        max_iterations = 500, 
        constraint_tolerance = 0.0001, 
        print_level=0, 
        lr=1,
        PCA=True,
        effect="ATT"):

        self.coefs = coefs
        self.max_iterations = max_iterations
        self.constraint_tolerance = constraint_tolerance
        self.print_level = print_level
        self.lr = lr
        self.PCA = PCA
        self.effect = effect
        if self.effect not in ['ATT', 'ATC', 'ATE']:
            sys.exit("Effect must be one of ATT, ATC, or ATE")


    def ebalance(
        self,
        Treatment,
        X,
        Y,
        base_weight=None):

        Treatment = np.asarray(Treatment)
        if self.effect == "ATC":
            Treatment = np.abs(Treatment-1) # revert treatment indicator so that the treatment group gets reweighted instead of the control group
            print("Estimating ATC:\n" + "-"*35)
        elif self.effect == "ATE":
            print("Estimating ATE:\n" + "-"*35)
        else:
            print("Estimating ATT:\n" + "-"*35)

        X = np.asarray(X)

        if np.isnan(Treatment).any():
           sys.exit("Treatment contains missing data")

        if np.var(Treatment) == 0:
            sys.exit("Treatment indicator ('Treatment') must contain both treatment and control observations")
        
        if np.isnan(X).any():
            sys.exit("X contains missing data")

        if not Treatment.shape[0] == X.shape[0]:
            sys.exit("length(Treatment) != nrow(X)")
        
        if not isinstance(self.max_iterations, int):
            sys.exit("length(max.iterations) != 1")

        if self.PCA:
            pca = PCA()
            X_c = X - X.mean(axis=0)
            X_c_pca = pca.fit_transform(X_c)
            X = X_c_pca[:,(pca.explained_variance_>=1) | (pca.explained_variance_ratio_>=0.00001)]
            print("PCA on X successful \n" + "-"*35)
        
        # set up elements
        ntreated  = np.sum(Treatment==1)
        ncontrols = np.sum(Treatment==0)

        if base_weight is None:
            base_weight = np.ones(ncontrols)
        elif not len(base_weight) == ncontrols:
            sys.exit("length(base_weight) !=  number of controls, sum(Treatment==0)")
        else:
            base_weight = np.asarray(base_weight)


        co_x = X[Treatment==0,:]
        co_x = np.column_stack((np.ones(ncontrols),co_x))

        if not np.linalg.matrix_rank(co_x) == co_x.shape[1]:
            sys.exit("collinearity in covariate matrix for controls (remove collinear covariates)")

        if self.effect == "ATE":
            tr_total = X.mean(axis=0)
        else:
            tr_total = X[Treatment==1,:].mean(axis=0)

        tr_total = np.insert(tr_total, 0, 1, axis=0)
        if self.coefs is None:
            self.coefs = np.insert(np.zeros(co_x.shape[1]-1), 0, np.log(1), axis=0)
        else:
            self.coefs = np.asarray(self.coefs)
           
        if not self.coefs.shape[0]==co_x.shape[1]:
            sys.exit("coefs needs to have same length as number of covariates plus one")

        if self.print_level >= 0:
            print(f"Set-up complete, balancing {co_x.shape[1]-1} covariates and 1 intercept\n")
        if self.effect != "ATE":
            print(f"Start finding weights for {ncontrols} units:\n" + "-"*35)
        
        if self.effect == "ATE":
            print(f"Start finding weights for control group with {ncontrols} units:\n" + "-"*35)
            weights = self._eb(tr_total, co_x, base_weight) #control group
            w = np.ones(X.shape[0])
            w[Treatment==0] = weights
            # treatment group
            print(f"Start finding weights for treatment group with {ntreated} units:\n" + "-"*35)
            base_weight = np.ones(ntreated)
            co_x = X[Treatment==1,:]
            co_x = np.column_stack((np.ones(ntreated),co_x))
            weights = self._eb(tr_total, co_x, base_weight)
            w[Treatment==1] = weights
        else:
            weights = self._eb(tr_total, co_x, base_weight)
            w = np.ones(X.shape[0])/ntreated
            w[Treatment==0] = weights
        
        if self.effect == "ATC":
            Treatment = np.abs(Treatment-1) # need to revert the treatment indicator to estimate atc

        wls_results = self._get_wls_results(se_type="HC2", Treatment=Treatment, Y=Y, weights=w)
        
        return {'converged': self.converged, 'maxdiff': self.maxdiff, 'w':w, "wls":wls_results}


    def _eb(self, tr_total, co_x, base_weight):
        self.converged = False
        for iter in range(self.max_iterations):
            weights_temp = np.exp(co_x.dot(self.coefs)) #(n, )
            weights_ebal = np.multiply(weights_temp, base_weight) #(n, )
            co_x_agg  = weights_ebal.dot(co_x) #(p, )
            gradient  = co_x_agg - tr_total
            self.maxdiff = max(np.absolute(gradient))
            if self.maxdiff < self.constraint_tolerance:
                self.converged = True
                print("algorithm has converged, final loss = " + str(self.maxdiff))
                break
            hessian = co_x.T.dot((co_x*weights_ebal[:, np.newaxis]))
            self.Coefs = self.coefs.copy()
            newton = np.linalg.solve(hessian, gradient)
            self.coefs -= newton*self.lr
            loss_new = self._line_searcher(ss=0, newton=newton, base_weight=base_weight, co_x=co_x, tr_total=tr_total, coefs=self.coefs)
            loss_old = self._line_searcher(ss=0, newton=newton, base_weight=base_weight, co_x=co_x, tr_total=tr_total, coefs=self.Coefs)

            if iter % 10==0 and self.print_level>=0:
                print("iteration = " + str(iter) + ", loss = " + str(loss_old))
                
            if loss_old <= loss_new:
                ss_min = minimize_scalar(self._line_searcher, bounds=(.0001, self.lr), args=(newton, base_weight, co_x, tr_total, self.Coefs), method='bounded')
                self.coefs = self.Coefs - ss_min.x*newton
        
        if self.converged == False:
            print("algorithm did not converged, final loss = " + str(self.maxdiff));

        return weights_ebal


    def _line_searcher(self, ss, newton, base_weight, co_x, tr_total, coefs):
        weights_temp = np.exp(co_x.dot((coefs-ss*newton)))
        weights_temp = np.multiply(weights_temp, base_weight)
        co_x_agg  = weights_temp.dot(co_x) #(p, )
        gradient  = co_x_agg - tr_total
        return max(np.absolute(gradient))

    
    def _get_wls_results(self, se_type, Treatment, Y, weights):
        t = sm.add_constant(Treatment.reshape(-1,1)) # intercept + treatment
        t = pd.DataFrame(data=t, columns=["const", "treatment"])
        mod_wls = sm.WLS(Y, t, weights=weights)
        res_wls = mod_wls.fit()
        return res_wls.get_robustcov_results(cov_type=se_type)


    def check_balance(self, X, Treatment, weights):
        weights[Treatment==1] = weights[Treatment==1]/np.sum(weights[Treatment==1])
        weights[Treatment==0] = weights[Treatment==0]/np.sum(weights[Treatment==0]) # normalize weights
        if self.effect == "ATC":
            Treatment = np.abs(Treatment-1)

        types = np.array([self._check_binary(X[x]) for x in X])
        col_names = np.array(list(X.columns.values))
        stds = np.std(X, axis=0)
        to_keep = np.std(X, axis=0)!=0
        types = types[to_keep]
        stds = stds[to_keep]
        col_drop = col_names[to_keep==False]
        col_names = col_names[to_keep]
        X = np.asarray(X)[:,to_keep]
        if self.effect == "ATE":
            tr_mean = np.mean(X, axis=0)
        else:
            tr_mean = np.dot(weights[Treatment==1], X[Treatment==1,:])
        before = np.round((tr_mean - np.mean(X[Treatment==0,:], axis=0))/stds, 2)
        after = np.round(tr_mean - np.dot(weights[Treatment==0], X[Treatment==0,:]), 2)
        out = {"Types": types, "Before_weighting": before, "After_weighting": after}
        print(pd.DataFrame(data=out, index=col_names).to_string())
        if len(col_drop)>0:
            print(f"\n*Note: Columns {col_drop} were dropped because their standard deviations are 0")


    def _check_binary(self, x):
        if len(set(x))==2:
            return "binary"
        else:
            return "cont"
        

class naive_ebal:
    def __init__(self,):
        pass
    def predict_effect(self, X, treatment, y, effect, max_iter=10):
        ebal_model = ebal_bin(
            effect = effect,
            PCA = True,
            print_level = -1
        )
        
        X, treatment, y = convert_pd_to_np(X, treatment, y)
        
        df_x = pd.DataFrame(X, columns = [f'x_{i}' for i in range(X.shape[1])])
        
        constraint_tolerance = 0.0001
        ebal_output = None
        for _ in range(max_iter):
            try:
                ebal_model = ebal_bin(
                    effect = effect,
                    PCA = True,
                    print_level = -1,
                    constraint_tolerance = constraint_tolerance,
                )
                ebal_output = ebal_model.ebalance(treatment, df_x, y)
                break
            except:
                constraint_tolerance = constraint_tolerance*10
        
        treatment_index = (treatment==1)
        control_index = (treatment==0)
        ebal_weight = ebal_output['w']

        mu1_hat = np.sum(y[treatment_index] * ebal_weight[treatment_index]) 
        mu0_hat = np.sum(y[control_index] * ebal_weight[control_index])

        return mu1_hat - mu0_hat
    def predict_att(self, X, treatment, y):
        return self.predict_effect(X, treatment, y, 'ATT')
        
    def predict_atc(self, X, treatment, y):
        return self.predict_effect(X, treatment, y, 'ATC')
    
    def predict_ate(self, X, treatment, y):
        return self.predict_effect(X, treatment, y, 'ATE')
    
    
# NN entropy balancing

def convert_pd_to_np(*args):
    output = [obj.to_numpy() if hasattr(obj, "to_numpy") else obj for obj in args]
    return output if len(output) > 1 else output[0]


def binary_classification_loss(concat_true, concat_pred):
    """
    Implements a classification (binary cross-entropy) loss function for DragonNet architecture.

    Args:
        - concat_true (tf.tensor): tensor of true samples, with shape (n_samples, 2)
                                   Each row in concat_true is comprised of (y, treatment)
        - concat_pred (tf.tensor): tensor of predictions, with shape (n_samples, 4)
                                   Each row in concat_pred is comprised of (y0, y1, propensity, epsilon)
    Returns:
        - (float): binary cross-entropy loss
    """
    t_true = concat_true[:, 1]
    t_pred = concat_pred[:, 2]
    t_pred = (t_pred + 0.000001) / 1.000002
    losst = tf.reduce_sum(keras_binary_crossentropy(t_true, t_pred))

    return losst


def regression_loss(concat_true, concat_pred):
    """
    Implements a regression (squared error) loss function for DragonNet architecture.

    Args:
        - concat_true (tf.tensor): tensor of true samples, with shape (n_samples, 2)
                                   Each row in concat_true is comprised of (y, treatment)
        - concat_pred (tf.tensor): tensor of predictions, with shape (n_samples, 4)
                                   Each row in concat_pred is comprised of (y0, y1, propensity, epsilon)
    Returns:
        - (float): aggregated regression loss
    """
    y_true = concat_true[:, 0]
    t_true = concat_true[:, 1]

    y0_pred = concat_pred[:, 0]
    y1_pred = concat_pred[:, 1]

    loss0 = tf.reduce_sum((1.0 - t_true) * tf.square(y_true - y0_pred))
    loss1 = tf.reduce_sum(t_true * tf.square(y_true - y1_pred))

    return loss0 + loss1

def binary_regression_loss(concat_true, concat_pred):
    """
    Implements a regression (squared error) loss function for DragonNet architecture.

    Args:
        - concat_true (tf.tensor): tensor of true samples, with shape (n_samples, 2)
                                   Each row in concat_true is comprised of (y, treatment)
        - concat_pred (tf.tensor): tensor of predictions, with shape (n_samples, 4)
                                   Each row in concat_pred is comprised of (y0, y1, propensity, epsilon)
    Returns:
        - (float): aggregated regression loss
    """
    y_true = concat_true[:, 0]
    t_true = concat_true[:, 1]

    y0_pred = concat_pred[:, 0]
    y1_pred = concat_pred[:, 1]
    
    y0_pred = (y0_pred + 0.000001) / 1.000002
    y1_pred = (y1_pred + 0.000001) / 1.000002

    loss0 = tf.reduce_sum((1.0 - t_true) * keras_binary_crossentropy(y_true, y0_pred))
    loss1 = tf.reduce_sum(t_true * keras_binary_crossentropy(y_true, y1_pred))

    return loss0 + loss1

def nn_loss(concat_true, concat_pred):
        """
        Implements regression + classification loss in one wrapper function.

        Args:
            - concat_true (tf.tensor): tensor of true samples, with shape (n_samples, 2)
                                    Each row in concat_true is comprised of (y, treatment)
            - concat_pred (tf.tensor): tensor of predictions, with shape (n_samples, 4)
                                    Each row in concat_pred is comprised of (y0, y1, propensity, epsilon)
        Returns:
            - (float): aggregated regression + classification loss
        """
        return regression_loss(concat_true, concat_pred) + binary_classification_loss(
            concat_true, concat_pred
        )
def nn_binary_loss(concat_true, concat_pred):
        """
        Implements regression + classification loss in one wrapper function.

        Args:
            - concat_true (tf.tensor): tensor of true samples, with shape (n_samples, 2)
                                    Each row in concat_true is comprised of (y, treatment)
            - concat_pred (tf.tensor): tensor of predictions, with shape (n_samples, 4)
                                    Each row in concat_pred is comprised of (y0, y1, propensity, epsilon)
        Returns:
            - (float): aggregated regression + classification loss
        """
        return binary_regression_loss(concat_true, concat_pred) + binary_classification_loss(
            concat_true, concat_pred
        )

def treatment_accuracy(concat_true, concat_pred):
    """
    Returns keras' binary_accuracy between treatment and prediction of propensity.

    Args:
        - concat_true (tf.tensor): tensor of true samples, with shape (n_samples, 2)
                                   Each row in concat_true is comprised of (y, treatment)
        - concat_pred (tf.tensor): tensor of predictions, with shape (n_samples, 4)
                                   Each row in concat_pred is comprised of (y0, y1, propensity, epsilon)
    Returns:
        - (float): binary accuracy
    """
    t_true = concat_true[:, 1]
    t_pred = concat_pred[:, 2]
    return binary_accuracy(t_true, t_pred)

    
class NNEbal:
    
    def __init__(self,user_params):
        
        if 'input_dim' not in user_params:
            raise Exception("input_dim must be specified!")
        
        params = {
            ## NN params
            'neurons_per_layer': 200, 
            'reg_l2': 0.5, 
            'verbose': True, 
            'val_split': 0.2, 
            'ratio': 1.0, 
            'batch_size': 64,
            'epochs': 300,
            'learning_rate': 1e-3, 
            'momentum': 0.9,
            'use_adam': True,
            'adam_epochs':30, 
            'adam_learning_rate': 1e-3,
            
            'act_fn': 'gelu',
            'num_layers': 5,
            'embedding_dim': 16,
            'dropout_rate': 0,
            
            'weighted_loss': True,
            
            'task': 'reg',
            # entropy params
            'PCA': True,
            'ebal_print_level': -1,
            # reproducibility
            'random_seed': None,
        }
        
        for k in params:
            params[k] = user_params.get(k,params[k])
                
        params['input_dim' ] = user_params['input_dim']
        self.params = params
        
        # Set random seeds for reproducibility
        if self.params['random_seed'] is not None:
            self._set_random_seeds(self.params['random_seed'])
       
        # neural network
        K.clear_session()
        
        inputs = Input(shape=(params['input_dim'],), name="input")
        
        # representation
        x = inputs
        for i in range(params['num_layers']):
            x = Dense(
                units=params['neurons_per_layer'],
                activation=params['act_fn'],
                kernel_initializer="RandomNormal",
                kernel_regularizer=regularizers.l2(params['reg_l2']),
            )(x)
        
        x = Dense(
            units = params['embedding_dim'],
            activation = params['act_fn'],
            kernel_initializer="RandomNormal",
            name = "embedding",
            kernel_regularizer=regularizers.l2(params['reg_l2']),
        )(x)
        
        embedding = Dropout(rate = params['dropout_rate'], name = 'embedding_dropout')(x)

        t_predictions = Dense(units=1, activation="sigmoid")(embedding)
        
        if params['task'] == 'reg':
            y_activation = None 
        elif params['task'] == 'class':
            y_activation = 'sigmoid'
        y0_predictions = Dense(
            units=1,
            activation=y_activation,
            name="y0_predictions",
        )(embedding)
        y1_predictions = Dense(
            units=1,
            activation=y_activation,
            name="y1_predictions",
        )(embedding)       
        
        concat_pred = Concatenate(1)(
            [y0_predictions, y1_predictions, t_predictions]
        )
        
        self.model = keras_Model(inputs=inputs, outputs=concat_pred)
        self.params = params
    
    def _set_random_seeds(self, seed):
        """
        Set random seeds for reproducibility across NumPy, Python random, and TensorFlow.
        
        Args:
            seed (int): Random seed value
        """
        np.random.seed(seed)
        random.seed(seed)
        tf.random.set_seed(seed)
        # For TensorFlow 2.x, also set global seed
        try:
            tf.keras.utils.set_random_seed(seed)
        except AttributeError:
            # Fallback for older TensorFlow versions
            pass
    
    def predict_ite(self, X):
        X = convert_pd_to_np(X)
        preds = self.model.predict(X)
        return (preds[:, 1] - preds[:, 0])
    
    def predict_effect(self, X, treatment, y, effect, max_iter=10):
        ebal_model = ebal_bin(
            effect = effect,
            PCA = self.params['PCA'],
            print_level = self.params['ebal_print_level']
        )
        
        X, treatment, y = convert_pd_to_np(X, treatment, y)
        embedding = self.embedding_model.predict(X)
        
        df_x = pd.DataFrame(embedding, columns = [f'x_{i}' for i in range(embedding.shape[1])])
        
        constraint_tolerance = 0.0001
        ebal_output = None
        for _ in range(max_iter):
            try:
                ebal_model = ebal_bin(
                    effect = effect,
                    PCA = self.params['PCA'],
                    print_level = self.params['ebal_print_level'],
                    constraint_tolerance = constraint_tolerance,
                )
                ebal_output = ebal_model.ebalance(treatment, df_x, y)
                break
            except:
                constraint_tolerance = constraint_tolerance*10
        
        treatment_index = (treatment==1)
        control_index = (treatment==0)
        ebal_weight = ebal_output['w']

        mu1_hat = np.sum(y[treatment_index] * ebal_weight[treatment_index]) 
        mu0_hat = np.sum(y[control_index] * ebal_weight[control_index])

        return mu1_hat - mu0_hat
    def predict_att(self, X, treatment, y, max_iter=10):
        return self.predict_effect(X, treatment, y, 'ATT', max_iter)
        
    def predict_atc(self, X, treatment, y, max_iter=10):
        return self.predict_effect(X, treatment, y, 'ATC', max_iter)
    
    def predict_ate(self, X, treatment, y, max_iter=10):
        return self.predict_effect(X, treatment, y, 'ATE', max_iter)
    
    def fit(self, X, treatment, y):
        X, treatment, y = convert_pd_to_np(X, treatment, y)
        y = np.hstack((y.reshape(-1, 1), treatment.reshape(-1, 1)))
              
        if self.params['weighted_loss']:
            u = np.mean(treatment)+1e-6  
            sample_w = treatment/2/u + (1-treatment)/2/(1-u)
        else:
            sample_w = np.array([1]*len(treatment))
            
        if self.params['task'] == 'reg':
            loss = nn_loss
            y_loss = regression_loss
        elif self.params['task'] == 'class':
            loss = nn_binary_loss
            y_loss = binary_regression_loss
            
        metrics = [
            y_loss,
            binary_classification_loss,
            treatment_accuracy,
            # track_epsilon,
        ]
        
        if 'use_adam' in self.params and self.params['use_adam']: 
            self.model.compile(
                optimizer=Adam(learning_rate=self.params['adam_learning_rate']), loss=loss, weighted_metrics=metrics
            )

            adam_callbacks = [
                TerminateOnNaN(),
                EarlyStopping(monitor="val_loss", patience=2, min_delta=0.0, restore_best_weights=True),
                ReduceLROnPlateau(
                    monitor="loss",
                    factor=0.5,
                    patience=5,
                    verbose=self.params['verbose'],
                    mode="auto",
                    min_delta=1e-8,
                    cooldown=0,
                    min_lr=0,
                ),
            ]
            
            self.model.fit(
                X,
                y,
                sample_weight = sample_w,
                callbacks=adam_callbacks,
                validation_split=self.params['val_split'],
                epochs=self.params['adam_epochs'],
                batch_size=self.params['batch_size'],
                verbose=self.params['verbose']
            )
        
        # 
        sgd_callbacks = [
            TerminateOnNaN(),
            EarlyStopping(monitor="val_loss", patience=40, min_delta=0.0, restore_best_weights=True),
            ReduceLROnPlateau(
                monitor="loss",
                factor=0.5,
                patience=5,
                verbose=self.params['verbose'],
                mode="auto",
                min_delta=0.0,
                cooldown=0,
                min_lr=0,
            ),
        ]
        self.model.compile(
            optimizer=SGD(learning_rate=self.params['learning_rate'], momentum=self.params['momentum'], 
                        nesterov=True, clipvalue=0.5),
            loss=loss,
            weighted_metrics=metrics,
        )
        self.model.fit(
            X,
            y,
            sample_weight = sample_w,
            callbacks=sgd_callbacks,
            validation_split=self.params['val_split'],
            epochs=self.params['epochs'],
            batch_size=self.params['batch_size'],
            verbose=self.params['verbose']
        )
       
        # Keep embedding model for entropy balancing
        embedding_model = keras_Model(
            inputs = self.model.input,
            outputs = self.model.get_layer('embedding').output
        )
        self.embedding_model = embedding_model
    

def download_url(url, save_path, chunk_size=128):
    print(">>> downloading ",url," into ",save_path,"...")
    r = requests.get(url, stream=True)
    with open(save_path, 'wb') as fd:
        for chunk in r.iter_content(chunk_size=chunk_size):
            fd.write(chunk)


class DataLoader(ABC):
    
    def __init__(self, spark = None):
        self.loaded = False  
        self.spark = spark
        self.sc = spark.sparkContext if spark else None

    @staticmethod
    def get_loader(dataset_name='IHDP'):
        if dataset_name=='IHDP':
            return IHDPLoader()
        elif dataset_name=='JOBS':
            return JOBSLoader()
        else:
            raise Exception('dataset not supported::'+str(dataset_name))
            
    @abstractmethod
    def load(self):
        pass 


    def split(self,X_df,W_df,Y_df,test_size=None,random_state=None):
        #split
        assert test_size is not None and test_size >0 and test_size < 1
        if random_state is not None:
            np.random.seed(random_state)
        msk = np.random.rand(len(X_df)) > test_size
        #
        X_df_tr = X_df[msk]
        W_df_tr = W_df[msk]
        Y_df_tr = Y_df[msk]
        #
        X_df_te = X_df[~msk]
        W_df_te = W_df[~msk]
        Y_df_te = Y_df[~msk]
        #
        return X_df_tr,W_df_tr,Y_df_tr, X_df_te,W_df_te,Y_df_te


class JOBSLoader(DataLoader):
    def __init__(self):
        super(JOBSLoader, self).__init__()

    def load(self):

        file_train = "data/jobs_DW_bin.new.10.train.npz"
        file_test = "data/jobs_DW_bin.new.10.test.npz"
        if not os.path.exists(file_train):
            download_url("https://www.fredjo.com/files/jobs_DW_bin.new.10.train.npz", file_train) 
        if not os.path.exists(file_test):
            download_url("https://www.fredjo.com/files/jobs_DW_bin.new.10.test.npz", file_test) 

        train_cv = np.load(file_train)
        test = np.load(file_test)
    
        self.X_tr    = train_cv.f.x.copy()
        self.T_tr    = train_cv.f.t.copy()
        self.Y_tr   = train_cv.f.yf.copy()
        self.E_tr = train_cv.f.e.copy()
        
        self.X_te    = test.f.x.copy()
        self.T_te    = test.f.t.copy()
        self.Y_te   = test.f.yf.copy()
        self.E_te  = test.f.e.copy()

        self.loaded = True
        
        return self.X_tr,self.T_tr, self.Y_tr, self.E_tr, \
            self.X_te,self.T_te, self.Y_te, self.E_te,
    
    
    def __len__(self):
        if not self.loaded:
            self.load()
        return self.x.shape[-1]
    
    def __getitem__(self, idx):
        if not self.loaded:
            self.load()
        return self.X_tr[:,:,idx],self.T_tr[:,idx], self.Y_tr[:,idx], self.E_tr[:,idx], \
            self.X_te[:,:,idx],self.T_te[:,idx], self.Y_te[:,idx], self.E_te[:,idx],
        
        

class IHDPLoader(DataLoader):
    
    def __init__(self):
        super(IHDPLoader, self).__init__()

    def load(self):
        file_train = "data/ihdp_npci_1-1000.train.npz"
        file_test = "data/ihdp_npci_1-1000.test.npz"
        if not os.path.exists(file_train):
            path_train_zip = 'tmp/ihdp_npci_1-1000.train.npz.zip'
            download_url("http://www.fredjo.com/files/ihdp_npci_1-1000.train.npz.zip", path_train_zip) 
            with zipfile.ZipFile(path_train_zip, 'r') as zip_ref:
                zip_ref.extractall('data')
        if not os.path.exists(file_test):
            path_test_zip = 'tmp/ihdp_npci_1-1000.test.npz.zip'
            download_url("http://www.fredjo.com/files/ihdp_npci_1-1000.test.npz.zip", path_test_zip) 
            with zipfile.ZipFile(path_train_zip, 'r') as zip_ref:
                zip_ref.extractall('data')
        # load 
        train_cv = np.load(file_train)
        test = np.load(file_test)
    
        self.X_tr    = train_cv.f.x.copy()
        self.T_tr    = train_cv.f.t.copy()
        self.YF_tr   = train_cv.f.yf.copy()
        self.YCF_tr  = train_cv.f.ycf.copy()
        self.mu_0_tr = train_cv.f.mu0.copy()
        self.mu_1_tr = train_cv.f.mu1.copy()
        
        self.X_te    = test.f.x.copy()
        self.T_te    = test.f.t.copy()
        self.YF_te   = test.f.yf.copy()
        self.YCF_te  = test.f.ycf.copy()
        self.mu_0_te = test.f.mu0.copy()
        self.mu_1_te = test.f.mu1.copy()
        
        self.loaded = True
        
        return self.X_tr,self.T_tr, self.YF_tr, self.YCF_tr, self.mu_0_tr, self.mu_1_tr, \
            self.X_te, self.T_te, self.YF_te, self.YCF_te, self.mu_0_te, self.mu_1_te
    
    
    def __len__(self):
        if not self.loaded:
            self.load()
        return self.X_tr.shape[-1]
    
    def __getitem__(self, idx):
        if not self.loaded:
            self.load()
        return self.X_tr[:,:,idx], self.T_tr[:,idx], self.YF_tr[:,idx], self.YCF_tr[:,idx],self.mu_0_tr[:,idx], self.mu_1_tr[:,idx], \
            self.X_te[:,:,idx], self.T_te[:,idx], self.YF_te[:,idx], self.YCF_te[:,idx], self.mu_0_te[:,idx], self.mu_1_te[:,idx]
        
        

            
            