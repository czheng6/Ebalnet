"""
Baseline methods for causal inference benchmarking.

Implements:
- IPW (Inverse Probability Weighting) with logistic regression
- TARNet (Treatment-Agnostic Representation Network)
- CFRNet WASS (Counterfactual Regression with Wasserstein regularization)
- Dragonnet

References:
- Shalit et al. (2017): Estimating individual treatment effect: generalization bounds and algorithms
- Shi et al. (2019): Adapting Neural Networks for the Estimation of Treatment Effects
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import backend as K
from tensorflow.keras import Input, Model
from tensorflow.keras.layers import Dense, Concatenate, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, TerminateOnNaN
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.optimizers.legacy import SGD
from tensorflow.keras.regularizers import l2
from sklearn.linear_model import LogisticRegression
import random


def convert_pd_to_np(*args):
    """Convert pandas objects to numpy arrays"""
    output = [obj.to_numpy() if hasattr(obj, "to_numpy") else obj for obj in args]
    return output if len(output) > 1 else output[0]


def set_random_seeds(seed):
    """Set random seeds for reproducibility"""
    np.random.seed(seed)
    random.seed(seed)
    tf.random.set_seed(seed)
    try:
        tf.keras.utils.set_random_seed(seed)
    except AttributeError:
        pass


# =====================================================
# IPW: Inverse Probability Weighting
# =====================================================

class IPW:
    """
    Inverse Probability Weighting estimator using logistic regression
    for propensity score estimation.
    """
    
    def __init__(self, regularization='l2', C=1.0, random_seed=None):
        """
        Args:
            regularization: Type of regularization ('l1', 'l2', 'elasticnet', or 'none')
            C: Inverse of regularization strength
            random_seed: Random seed for reproducibility
        """
        self.regularization = regularization
        self.C = C
        self.random_seed = random_seed
        self.propensity_model = None
        
    def fit(self, X, treatment, y):
        """Fit propensity score model"""
        X, treatment, y = convert_pd_to_np(X, treatment, y)
        
        if self.random_seed is not None:
            set_random_seeds(self.random_seed)
        
        # Fit logistic regression for propensity score
        if self.regularization == 'none':
            self.propensity_model = LogisticRegression(
                penalty=None,
                max_iter=1000,
                random_state=self.random_seed
            )
        else:
            self.propensity_model = LogisticRegression(
                penalty=self.regularization,
                C=self.C,
                max_iter=1000,
                random_state=self.random_seed,
                solver='saga' if self.regularization == 'elasticnet' else 'lbfgs'
            )
        
        self.propensity_model.fit(X, treatment)
        
    def predict_propensity(self, X):
        """Predict propensity scores"""
        X = convert_pd_to_np(X)
        return self.propensity_model.predict_proba(X)[:, 1]
    
    def predict_att(self, X, treatment, y, clip_bounds=(0.01, 0.99)):
        """Estimate ATT using IPW"""
        X, treatment, y = convert_pd_to_np(X, treatment, y)
        
        e = self.predict_propensity(X)
        e = np.clip(e, clip_bounds[0], clip_bounds[1])
        
        # ATT weights
        n1 = np.sum(treatment)
        treated_mean = np.sum(y * treatment) / n1
        
        # Weighted control mean
        weights = e / (1 - e)
        control_weighted_mean = np.sum(y * (1 - treatment) * weights) / np.sum((1 - treatment) * weights)
        
        return treated_mean - control_weighted_mean
    
    def predict_ate(self, X, treatment, y, clip_bounds=(0.01, 0.99)):
        """Estimate ATE using IPW"""
        X, treatment, y = convert_pd_to_np(X, treatment, y)
        
        e = self.predict_propensity(X)
        e = np.clip(e, clip_bounds[0], clip_bounds[1])
        
        # Horvitz-Thompson estimator
        n = len(y)
        ate = np.mean(y * treatment / e - y * (1 - treatment) / (1 - e))
        
        return ate


# =====================================================
# TARNet: Treatment-Agnostic Representation Network
# =====================================================

class TARNet:
    """
    TARNet from Shalit et al. (2017).
    Learns shared representation with separate outcome heads for treated/control.
    """
    
    def __init__(self, input_dim, hidden_layers=3, hidden_units=200, 
                 repr_dim=100, reg_l2=0.01, dropout_rate=0.0,
                 learning_rate=1e-3, epochs=300, batch_size=64,
                 val_split=0.2, verbose=False, random_seed=None):
        
        self.input_dim = input_dim
        self.hidden_layers = hidden_layers
        self.hidden_units = hidden_units
        self.repr_dim = repr_dim
        self.reg_l2 = reg_l2
        self.dropout_rate = dropout_rate
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size
        self.val_split = val_split
        self.verbose = verbose
        self.random_seed = random_seed
        
        if self.random_seed is not None:
            set_random_seeds(self.random_seed)
        
        self._build_model()
    
    def _build_model(self):
        """Build TARNet architecture"""
        K.clear_session()
        
        inputs = Input(shape=(self.input_dim,), name="input")
        
        # Shared representation layers
        x = inputs
        for i in range(self.hidden_layers):
            x = Dense(
                units=self.hidden_units,
                activation='elu',
                kernel_regularizer=l2(self.reg_l2),
                name=f'shared_{i}'
            )(x)
            if self.dropout_rate > 0:
                x = Dropout(self.dropout_rate)(x)
        
        # Representation layer
        repr_layer = Dense(
            units=self.repr_dim,
            activation='elu',
            kernel_regularizer=l2(self.reg_l2),
            name='representation'
        )(x)
        
        # Outcome head for control (T=0)
        y0_hidden = repr_layer
        for i in range(2):
            y0_hidden = Dense(
                units=self.hidden_units // 2,
                activation='elu',
                kernel_regularizer=l2(self.reg_l2),
                name=f'y0_hidden_{i}'
            )(y0_hidden)
        y0_pred = Dense(units=1, activation=None, name='y0_pred')(y0_hidden)
        
        # Outcome head for treated (T=1)
        y1_hidden = repr_layer
        for i in range(2):
            y1_hidden = Dense(
                units=self.hidden_units // 2,
                activation='elu',
                kernel_regularizer=l2(self.reg_l2),
                name=f'y1_hidden_{i}'
            )(y1_hidden)
        y1_pred = Dense(units=1, activation=None, name='y1_pred')(y1_hidden)
        
        # Concatenate outputs: [y0, y1]
        concat_pred = Concatenate(axis=1)([y0_pred, y1_pred])
        
        self.model = Model(inputs=inputs, outputs=concat_pred)
        
    def _tarnet_loss(self, concat_true, concat_pred):
        """TARNet loss: factual outcome regression"""
        y_true = concat_true[:, 0]
        t_true = concat_true[:, 1]
        
        y0_pred = concat_pred[:, 0]
        y1_pred = concat_pred[:, 1]
        
        # Only train on factual outcomes
        loss0 = tf.reduce_mean((1.0 - t_true) * tf.square(y_true - y0_pred))
        loss1 = tf.reduce_mean(t_true * tf.square(y_true - y1_pred))
        
        return loss0 + loss1
    
    def fit(self, X, treatment, y):
        """Train the model"""
        X, treatment, y = convert_pd_to_np(X, treatment, y)
        
        # Format targets: [y, t]
        y_concat = np.column_stack([y, treatment])
        
        self.model.compile(
            optimizer=Adam(learning_rate=self.learning_rate),
            loss=self._tarnet_loss
        )
        
        callbacks = [
            TerminateOnNaN(),
            EarlyStopping(monitor='val_loss', patience=30, restore_best_weights=True),
            ReduceLROnPlateau(monitor='loss', factor=0.5, patience=10, min_lr=1e-6)
        ]
        
        self.model.fit(
            X, y_concat,
            validation_split=self.val_split,
            epochs=self.epochs,
            batch_size=self.batch_size,
            callbacks=callbacks,
            verbose=self.verbose
        )
    
    def predict(self, X):
        """Predict potential outcomes [y0, y1]"""
        X = convert_pd_to_np(X)
        return self.model.predict(X, verbose=0)
    
    def predict_ite(self, X):
        """Predict individual treatment effect"""
        preds = self.predict(X)
        return preds[:, 1] - preds[:, 0]
    
    def predict_att(self, X, treatment, y):
        """Estimate ATT"""
        X, treatment, y = convert_pd_to_np(X, treatment, y)
        preds = self.predict(X)
        
        # For treated units, estimate counterfactual control outcome
        treated_mask = treatment == 1
        att = np.mean(y[treated_mask] - preds[treated_mask, 0])
        
        return att
    
    def predict_ate(self, X, treatment, y):
        """Estimate ATE"""
        X = convert_pd_to_np(X)
        preds = self.predict(X)
        return np.mean(preds[:, 1] - preds[:, 0])


# =====================================================
# CFRNet WASS: CFR with Wasserstein Distance Regularization
# =====================================================

def wasserstein_distance_approx(X_repr, treatment, p=2):
    """
    Approximate Wasserstein distance between treated and control representations.
    Uses sorting-based approximation for 1D case, extended to multi-dim.
    """
    treated_mask = tf.cast(treatment > 0.5, tf.float32)
    control_mask = 1.0 - treated_mask
    
    n_treated = tf.reduce_sum(treated_mask) + 1e-8
    n_control = tf.reduce_sum(control_mask) + 1e-8
    
    # Compute means
    treated_mean = tf.reduce_sum(X_repr * tf.expand_dims(treated_mask, 1), axis=0) / n_treated
    control_mean = tf.reduce_sum(X_repr * tf.expand_dims(control_mask, 1), axis=0) / n_control
    
    # Compute variances
    treated_var = tf.reduce_sum(
        tf.square(X_repr - treated_mean) * tf.expand_dims(treated_mask, 1), axis=0
    ) / n_treated
    control_var = tf.reduce_sum(
        tf.square(X_repr - control_mean) * tf.expand_dims(control_mask, 1), axis=0
    ) / n_control
    
    # 2-Wasserstein approximation using moment matching
    mean_diff = tf.reduce_sum(tf.square(treated_mean - control_mean))
    var_diff = tf.reduce_sum(tf.square(tf.sqrt(treated_var + 1e-8) - tf.sqrt(control_var + 1e-8)))
    
    return mean_diff + var_diff


class CFRNet_WASS:
    """
    Counterfactual Regression Network with Wasserstein regularization.
    From Shalit et al. (2017).
    """
    
    def __init__(self, input_dim, hidden_layers=3, hidden_units=200,
                 repr_dim=100, reg_l2=0.01, alpha=1.0, dropout_rate=0.0,
                 learning_rate=1e-3, epochs=300, batch_size=64,
                 val_split=0.2, verbose=False, random_seed=None):
        """
        Args:
            alpha: Weight for Wasserstein regularization (IPM penalty)
        """
        self.input_dim = input_dim
        self.hidden_layers = hidden_layers
        self.hidden_units = hidden_units
        self.repr_dim = repr_dim
        self.reg_l2 = reg_l2
        self.alpha = alpha
        self.dropout_rate = dropout_rate
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size
        self.val_split = val_split
        self.verbose = verbose
        self.random_seed = random_seed
        
        if self.random_seed is not None:
            set_random_seeds(self.random_seed)
        
        self._build_model()
    
    def _build_model(self):
        """Build CFRNet architecture"""
        K.clear_session()
        
        inputs = Input(shape=(self.input_dim,), name="input")
        
        # Shared representation layers
        x = inputs
        for i in range(self.hidden_layers):
            x = Dense(
                units=self.hidden_units,
                activation='elu',
                kernel_regularizer=l2(self.reg_l2),
                name=f'shared_{i}'
            )(x)
            if self.dropout_rate > 0:
                x = Dropout(self.dropout_rate)(x)
        
        # Representation layer
        repr_layer = Dense(
            units=self.repr_dim,
            activation='elu',
            kernel_regularizer=l2(self.reg_l2),
            name='representation'
        )(x)
        
        # Outcome head for control (T=0)
        y0_hidden = repr_layer
        for i in range(2):
            y0_hidden = Dense(
                units=self.hidden_units // 2,
                activation='elu',
                kernel_regularizer=l2(self.reg_l2),
                name=f'y0_hidden_{i}'
            )(y0_hidden)
        y0_pred = Dense(units=1, activation=None, name='y0_pred')(y0_hidden)
        
        # Outcome head for treated (T=1)
        y1_hidden = repr_layer
        for i in range(2):
            y1_hidden = Dense(
                units=self.hidden_units // 2,
                activation='elu',
                kernel_regularizer=l2(self.reg_l2),
                name=f'y1_hidden_{i}'
            )(y1_hidden)
        y1_pred = Dense(units=1, activation=None, name='y1_pred')(y1_hidden)
        
        # Output: [y0, y1, repr (flattened for regularization)]
        concat_pred = Concatenate(axis=1)([y0_pred, y1_pred, repr_layer])
        
        self.model = Model(inputs=inputs, outputs=concat_pred)
        self.repr_dim_actual = self.repr_dim
        
    def _cfrnet_loss(self, concat_true, concat_pred):
        """CFRNet loss with Wasserstein regularization"""
        y_true = concat_true[:, 0]
        t_true = concat_true[:, 1]
        
        y0_pred = concat_pred[:, 0]
        y1_pred = concat_pred[:, 1]
        repr_layer = concat_pred[:, 2:]  # Representation
        
        # Factual loss
        loss0 = tf.reduce_mean((1.0 - t_true) * tf.square(y_true - y0_pred))
        loss1 = tf.reduce_mean(t_true * tf.square(y_true - y1_pred))
        factual_loss = loss0 + loss1
        
        # Wasserstein regularization
        wass_dist = wasserstein_distance_approx(repr_layer, t_true)
        
        return factual_loss + self.alpha * wass_dist
    
    def fit(self, X, treatment, y):
        """Train the model"""
        X, treatment, y = convert_pd_to_np(X, treatment, y)
        
        y_concat = np.column_stack([y, treatment])
        
        self.model.compile(
            optimizer=Adam(learning_rate=self.learning_rate),
            loss=self._cfrnet_loss
        )
        
        callbacks = [
            TerminateOnNaN(),
            EarlyStopping(monitor='val_loss', patience=30, restore_best_weights=True),
            ReduceLROnPlateau(monitor='loss', factor=0.5, patience=10, min_lr=1e-6)
        ]
        
        self.model.fit(
            X, y_concat,
            validation_split=self.val_split,
            epochs=self.epochs,
            batch_size=self.batch_size,
            callbacks=callbacks,
            verbose=self.verbose
        )
    
    def predict(self, X):
        """Predict potential outcomes [y0, y1]"""
        X = convert_pd_to_np(X)
        preds = self.model.predict(X, verbose=0)
        return preds[:, :2]  # Only return y0, y1
    
    def predict_ite(self, X):
        """Predict individual treatment effect"""
        preds = self.predict(X)
        return preds[:, 1] - preds[:, 0]
    
    def predict_att(self, X, treatment, y):
        """Estimate ATT"""
        X, treatment, y = convert_pd_to_np(X, treatment, y)
        preds = self.predict(X)
        
        treated_mask = treatment == 1
        att = np.mean(y[treated_mask] - preds[treated_mask, 0])
        
        return att
    
    def predict_ate(self, X, treatment, y):
        """Estimate ATE"""
        X = convert_pd_to_np(X)
        preds = self.predict(X)
        return np.mean(preds[:, 1] - preds[:, 0])


# =====================================================
# Dragonnet
# =====================================================

class Dragonnet:
    """
    Dragonnet from Shi et al. (2019).
    Three-headed network: two outcome heads + propensity head.
    Uses targeted regularization for double robustness.
    """
    
    def __init__(self, input_dim, hidden_layers=3, hidden_units=200,
                 repr_dim=100, reg_l2=0.01, dropout_rate=0.0,
                 ratio=1.0, targeted_reg=True, epsilon_weight=0.01,
                 learning_rate=1e-3, epochs=300, batch_size=64,
                 val_split=0.2, verbose=False, random_seed=None):
        """
        Args:
            ratio: Weight for propensity loss vs outcome loss
            targeted_reg: Whether to use targeted regularization
            epsilon_weight: Weight for targeted regularization term
        """
        self.input_dim = input_dim
        self.hidden_layers = hidden_layers
        self.hidden_units = hidden_units
        self.repr_dim = repr_dim
        self.reg_l2 = reg_l2
        self.dropout_rate = dropout_rate
        self.ratio = ratio
        self.targeted_reg = targeted_reg
        self.epsilon_weight = epsilon_weight
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size
        self.val_split = val_split
        self.verbose = verbose
        self.random_seed = random_seed
        
        if self.random_seed is not None:
            set_random_seeds(self.random_seed)
        
        self._build_model()
    
    def _build_model(self):
        """Build Dragonnet architecture"""
        K.clear_session()
        
        inputs = Input(shape=(self.input_dim,), name="input")
        
        # Shared representation layers
        x = inputs
        for i in range(self.hidden_layers):
            x = Dense(
                units=self.hidden_units,
                activation='elu',
                kernel_regularizer=l2(self.reg_l2),
                name=f'shared_{i}'
            )(x)
            if self.dropout_rate > 0:
                x = Dropout(self.dropout_rate)(x)
        
        # Representation layer
        repr_layer = Dense(
            units=self.repr_dim,
            activation='elu',
            kernel_regularizer=l2(self.reg_l2),
            name='representation'
        )(x)
        
        # Propensity head
        t_hidden = Dense(
            units=self.hidden_units // 2,
            activation='elu',
            kernel_regularizer=l2(self.reg_l2),
            name='t_hidden'
        )(repr_layer)
        t_pred = Dense(units=1, activation='sigmoid', name='t_pred')(t_hidden)
        
        # Outcome head for control (T=0)
        y0_hidden = repr_layer
        for i in range(2):
            y0_hidden = Dense(
                units=self.hidden_units // 2,
                activation='elu',
                kernel_regularizer=l2(self.reg_l2),
                name=f'y0_hidden_{i}'
            )(y0_hidden)
        y0_pred = Dense(units=1, activation=None, name='y0_pred')(y0_hidden)
        
        # Outcome head for treated (T=1)
        y1_hidden = repr_layer
        for i in range(2):
            y1_hidden = Dense(
                units=self.hidden_units // 2,
                activation='elu',
                kernel_regularizer=l2(self.reg_l2),
                name=f'y1_hidden_{i}'
            )(y1_hidden)
        y1_pred = Dense(units=1, activation=None, name='y1_pred')(y1_hidden)
        
        # Epsilon for targeted regularization
        epsilon = Dense(units=1, activation=None, name='epsilon')(repr_layer)
        
        # Concatenate outputs: [y0, y1, t, epsilon]
        concat_pred = Concatenate(axis=1)([y0_pred, y1_pred, t_pred, epsilon])
        
        self.model = Model(inputs=inputs, outputs=concat_pred)
    
    def _dragonnet_loss(self, concat_true, concat_pred):
        """Dragonnet loss with optional targeted regularization"""
        y_true = concat_true[:, 0]
        t_true = concat_true[:, 1]
        
        y0_pred = concat_pred[:, 0]
        y1_pred = concat_pred[:, 1]
        t_pred = concat_pred[:, 2]
        epsilon = concat_pred[:, 3]
        
        # Clip propensity for numerical stability
        t_pred_clipped = tf.clip_by_value(t_pred, 1e-6, 1 - 1e-6)
        
        # Outcome loss (factual)
        loss0 = tf.reduce_mean((1.0 - t_true) * tf.square(y_true - y0_pred))
        loss1 = tf.reduce_mean(t_true * tf.square(y_true - y1_pred))
        outcome_loss = loss0 + loss1
        
        # Propensity loss (binary cross-entropy)
        propensity_loss = tf.reduce_mean(
            -t_true * tf.math.log(t_pred_clipped) 
            - (1 - t_true) * tf.math.log(1 - t_pred_clipped)
        )
        
        total_loss = outcome_loss + self.ratio * propensity_loss
        
        # Targeted regularization
        if self.targeted_reg:
            y_pred = t_true * y1_pred + (1 - t_true) * y0_pred
            h = t_true / t_pred_clipped - (1 - t_true) / (1 - t_pred_clipped)
            
            y_pert = y_pred + epsilon * h
            targeted_loss = tf.reduce_mean(tf.square(y_true - y_pert))
            total_loss = total_loss + self.epsilon_weight * targeted_loss
        
        return total_loss
    
    def fit(self, X, treatment, y):
        """Train the model"""
        X, treatment, y = convert_pd_to_np(X, treatment, y)
        
        y_concat = np.column_stack([y, treatment])
        
        self.model.compile(
            optimizer=Adam(learning_rate=self.learning_rate),
            loss=self._dragonnet_loss
        )
        
        callbacks = [
            TerminateOnNaN(),
            EarlyStopping(monitor='val_loss', patience=30, restore_best_weights=True),
            ReduceLROnPlateau(monitor='loss', factor=0.5, patience=10, min_lr=1e-6)
        ]
        
        self.model.fit(
            X, y_concat,
            validation_split=self.val_split,
            epochs=self.epochs,
            batch_size=self.batch_size,
            callbacks=callbacks,
            verbose=self.verbose
        )
    
    def predict(self, X):
        """Predict potential outcomes and propensity [y0, y1, t, epsilon]"""
        X = convert_pd_to_np(X)
        return self.model.predict(X, verbose=0)
    
    def predict_ite(self, X):
        """Predict individual treatment effect"""
        preds = self.predict(X)
        return preds[:, 1] - preds[:, 0]
    
    def predict_propensity(self, X):
        """Predict propensity scores"""
        preds = self.predict(X)
        return preds[:, 2]
    
    def predict_att(self, X, treatment, y):
        """Estimate ATT using AIPW"""
        X, treatment, y = convert_pd_to_np(X, treatment, y)
        preds = self.predict(X)
        
        y0_pred = preds[:, 0]
        y1_pred = preds[:, 1]
        e = np.clip(preds[:, 2], 0.01, 0.99)
        
        # AIPW estimator for ATT
        treated_mask = treatment == 1
        n1 = np.sum(treated_mask)
        
        # Direct estimate for treated outcomes
        mu1 = np.mean(y[treated_mask])
        
        # AIPW for counterfactual control outcome
        weights = e / (1 - e)
        weighted_resid = weights * (y - y0_pred) * (1 - treatment)
        mu0 = np.mean(y0_pred[treated_mask]) + np.sum(weighted_resid) / n1
        
        return mu1 - mu0
    
    def predict_att_simple(self, X, treatment, y):
        """Simple ATT estimate (plug-in)"""
        X, treatment, y = convert_pd_to_np(X, treatment, y)
        preds = self.predict(X)
        
        treated_mask = treatment == 1
        att = np.mean(y[treated_mask] - preds[treated_mask, 0])
        
        return att
    
    def predict_ate(self, X, treatment, y):
        """Estimate ATE"""
        X = convert_pd_to_np(X)
        preds = self.predict(X)
        return np.mean(preds[:, 1] - preds[:, 0])


# =====================================================
# Convenience functions for evaluation
# =====================================================

def evaluate_method_ihdp(method, X, treatment, y, mu0, mu1, effect='ATT'):
    """
    Evaluate a method on IHDP-style data with known ground truth.
    
    Args:
        method: Fitted causal inference method
        X: Covariates
        treatment: Treatment indicator
        y: Observed outcome
        mu0, mu1: True conditional means for Y(0) and Y(1)
        effect: 'ATT' or 'ATE'
    
    Returns:
        error: Absolute error in treatment effect estimation
    """
    if effect == 'ATT':
        # True ATT
        truth = np.mean(mu1[treatment == 1] - mu0[treatment == 1])
        # Estimated ATT
        if hasattr(method, 'predict_att'):
            estimate = method.predict_att(X, treatment, y)
        else:
            raise ValueError("Method does not have predict_att")
    elif effect == 'ATE':
        # True ATE
        truth = np.mean(mu1 - mu0)
        # Estimated ATE
        if hasattr(method, 'predict_ate'):
            estimate = method.predict_ate(X, treatment, y)
        else:
            raise ValueError("Method does not have predict_ate")
    else:
        raise ValueError("effect must be 'ATT' or 'ATE'")
    
    return np.abs(truth - estimate)


def evaluate_method_jobs(method, X, treatment, y, e_indicator, effect='ATT'):
    """
    Evaluate a method on JOBS data.
    Ground truth ATT is computed from randomized experimental sample (e=1).
    
    Args:
        method: Fitted causal inference method
        X: Covariates
        treatment: Treatment indicator
        y: Observed outcome
        e_indicator: Indicator for experimental sample
        effect: 'ATT'
    
    Returns:
        error: Absolute error in treatment effect estimation
    """
    # True ATT from experimental sample
    exp_mask = e_indicator == 1
    exp_treated = (treatment == 1) & exp_mask
    exp_control = (treatment == 0) & exp_mask
    
    truth = np.mean(y[exp_treated]) - np.mean(y[exp_control])
    
    # Estimated ATT
    if hasattr(method, 'predict_att'):
        estimate = method.predict_att(X, treatment, y)
    else:
        raise ValueError("Method does not have predict_att")
    
    return np.abs(truth - estimate)


if __name__ == "__main__":
    # Quick test
    print("Testing baseline methods...")
    
    # Generate synthetic data
    np.random.seed(42)
    n = 1000
    p = 10
    
    X = np.random.randn(n, p)
    e = 1 / (1 + np.exp(-X[:, 0] - 0.5 * X[:, 1]))
    treatment = (np.random.rand(n) < e).astype(float)
    
    y0 = X[:, 0] + X[:, 1] + np.random.randn(n) * 0.5
    y1 = y0 + 2 + X[:, 2]  # Treatment effect = 2 + X[:, 2]
    y = treatment * y1 + (1 - treatment) * y0
    
    true_att = np.mean(y1[treatment == 1] - y0[treatment == 1])
    true_ate = np.mean(y1 - y0)
    
    print(f"True ATT: {true_att:.3f}")
    print(f"True ATE: {true_ate:.3f}")
    
    # Test IPW
    print("\n--- IPW ---")
    ipw = IPW(random_seed=42)
    ipw.fit(X, treatment, y)
    print(f"IPW ATT: {ipw.predict_att(X, treatment, y):.3f}")
    print(f"IPW ATE: {ipw.predict_ate(X, treatment, y):.3f}")
    
    # Test TARNet
    print("\n--- TARNet ---")
    tarnet = TARNet(input_dim=p, epochs=50, verbose=False, random_seed=42)
    tarnet.fit(X, treatment, y)
    print(f"TARNet ATT: {tarnet.predict_att(X, treatment, y):.3f}")
    print(f"TARNet ATE: {tarnet.predict_ate(X, treatment, y):.3f}")
    
    # Test CFRNet WASS
    print("\n--- CFRNet WASS ---")
    cfrnet = CFRNet_WASS(input_dim=p, alpha=1.0, epochs=50, verbose=False, random_seed=42)
    cfrnet.fit(X, treatment, y)
    print(f"CFRNet WASS ATT: {cfrnet.predict_att(X, treatment, y):.3f}")
    print(f"CFRNet WASS ATE: {cfrnet.predict_ate(X, treatment, y):.3f}")
    
    # Test Dragonnet
    print("\n--- Dragonnet ---")
    dragonnet = Dragonnet(input_dim=p, epochs=50, verbose=False, random_seed=42)
    dragonnet.fit(X, treatment, y)
    print(f"Dragonnet ATT: {dragonnet.predict_att(X, treatment, y):.3f}")
    print(f"Dragonnet ATE: {dragonnet.predict_ate(X, treatment, y):.3f}")
    
    print("\nAll tests passed!")

