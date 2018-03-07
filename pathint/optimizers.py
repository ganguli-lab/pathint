# Copyright (c) 2017 Ben Poole & Friedemann Zenke
# MIT License -- see LICENSE for details
# 
# This file is part of the code to reproduce the core results of:
# Zenke, F., Poole, B., and Ganguli, S. (2017). Continual Learning Through
# Synaptic Intelligence. In Proceedings of the 34th International Conference on
# Machine Learning, D. Precup, and Y.W. Teh, eds. (International Convention
# Centre, Sydney, Australia: PMLR), pp. 3987-3995.
# http://proceedings.mlr.press/v70/zenke17a.html
#
"""Optimization algorithms."""

import tensorflow as tf

import numpy as np
import keras
from keras import backend as K
from keras.optimizers import Optimizer
from keras.callbacks import Callback
from pathint.utils import extract_weight_changes, compute_updates
from pathint.regularizers import quadratic_regularizer
from collections import OrderedDict


class KOOptimizer(Optimizer):
    """An optimizer whose loss depends on its own updates."""

    def _allocate_var(self, name=None):
        return {w: K.zeros(w.get_shape(), name=name) for w in self.weights}

    def _allocate_vars(self, names):
        #TODO: add names, better shape/init checking
        self.vars = {name: self._allocate_var(name=name) for name in names}

    def __init__(self, opt, step_updates=[], task_updates=[], init_updates=[], task_metrics = {}, regularizer_fn=quadratic_regularizer,
                lam=1.0, model=None, compute_average_loss=False, compute_average_weights=False, **kwargs):
        """Instantiate an optimzier that depends on its own updates.

        Args:
            opt: Keras optimizer
            step_updates: OrderedDict or List of tuples
                Contains variable names and updates to be run at each step:
                (name, lambda vars, weight, prev_val: new_val). See below for details.
            task_updates:  same as step_updates but run after each task
            init_updates: updates to be run before using the optimizer
            task_metrics: list of names of metrics to compute on full data/unionset after a task
            regularizer_fn (optional): function, takes in weights and variables returns scalar
                defaults to EWC regularizer
            lam: scalar penalty that multiplies the regularization term
            model: Keras model to be optimized. Needed to compute Fisher information
            compute_average_loss: compute EMA of the loss, default: False
            compute_average_weights: compute EMA of the weights, default: False

        Variables are created for each name in the task and step updates. Note that you cannot
        use the name 'grads', 'unreg_grads' or 'deltas' as those are reserved to contain the gradients
        of the full loss, loss without regularization, and the weight updates at each step.
        You can access them in the vars dict, e.g.: oopt.vars['grads']

        The step and task update functions have the signature:
            def update_fn(vars, weight, prev_val):
                '''Compute the new value for a variable.
                Args:
                    vars: optimization variables (OuroborosOptimzier.vars)
                    weight: weight Variable in model that this variable is associated with.
                    prev_val: previous value of this varaible
                Returns:
                    Tensor representing the new value'''

        You can run both task and step updates on the same variable, allowing you to reset
        step variables after each task.
        """
        super(KOOptimizer, self).__init__(**kwargs)
        if not isinstance(opt, keras.optimizers.Optimizer):
            raise ValueError("opt must be an instance of keras.optimizers.Optimizer but got %s"%type(opt))
        if not isinstance(step_updates, OrderedDict):
            step_updates = OrderedDict(step_updates)
        if not isinstance(task_updates, OrderedDict): task_updates = OrderedDict(task_updates)
        if not isinstance(init_updates, OrderedDict): init_updates = OrderedDict(init_updates)
        # task_metrics
        self.names = set().union(step_updates.keys(), task_updates.keys(), task_metrics.keys())
        if 'grads' in self.names or 'deltas' in self.names:
            raise ValueError("Optimization variables cannot be named 'grads' or 'deltas'")
        self.step_updates = step_updates
        self.task_updates = task_updates
        self.init_updates = init_updates
        self.compute_average_loss = compute_average_loss
        self.regularizer_fn = regularizer_fn
        # Compute loss and gradients
        self.lam = K.variable(value=lam, dtype=tf.float32, name="lam")
        self.nb_data = K.variable(value=1.0, dtype=tf.float32, name="nb_data")
        self.opt = opt
        #self.compute_fisher = compute_fisher
        #if compute_fisher and model is None:
        #    raise ValueError("To compute Fisher information, you need to pass in a Keras model object ")
        self.model = model
        self.task_metrics = task_metrics
        self.compute_average_weights = compute_average_weights

    def set_strength(self, val):
        K.set_value(self.lam, val)

    def set_nb_data(self, nb):
        K.set_value(self.nb_data, nb)

    def get_updates(self, weights, constraints, initial_loss, model=None):
        self.weights = weights
        # Allocate variables
        with tf.variable_scope("KOOptimizer"):
            self._allocate_vars(self.names)

        #grads = self.get_gradients(loss, params)

        # Compute loss and gradients
        self.regularizer = 0.0 if self.regularizer_fn is None else self.regularizer_fn(weights, self.vars)
        self.initial_loss = initial_loss
        self.loss = initial_loss + self.lam * self.regularizer
        with tf.variable_scope("wrapped_optimizer"):
            self._weight_update_op, self._grads, self._deltas = compute_updates(self.opt, self.loss, weights)

        wrapped_opt_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, "wrapped_optimizer")
        self.init_opt_vars = tf.variables_initializer(wrapped_opt_vars)

        self.vars['unreg_grads'] = dict(zip(weights, tf.gradients(self.initial_loss, weights)))
        # Compute updates
        self.vars['grads'] = dict(zip(weights, self._grads))
        self.vars['deltas'] = dict(zip(weights, self._deltas))
        # Keep a pointer to self in vars so we can use it in the updates
        self.vars['oopt'] = self
        # Keep number of data samples handy for normalization purposes
        self.vars['nb_data'] = self.nb_data

        if self.compute_average_weights:
            with tf.variable_scope("weight_emga") as scope:
                weight_ema = tf.train.ExponentialMovingAverage(decay=0.99, zero_debias=True)
                self.maintain_weight_averages_op = weight_ema.apply(self.weights)
                self.vars['average_weights'] = {w: weight_ema.average(w) for w in self.weights}
            self.weight_ema_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=scope.name)
            self.init_weight_ema_vars = tf.variables_initializer(self.weight_ema_vars)
            print(">>>>>")
            K.get_session().run(self.init_weight_ema_vars)
        if self.compute_average_loss:
            with tf.variable_scope("ema") as scope:
                ema = tf.train.ExponentialMovingAverage(decay=0.99, zero_debias=True)
                self.maintain_averages_op = ema.apply([self.initial_loss])
                self.ema_loss = ema.average(self.initial_loss)
                self.prev_loss = tf.Variable(0.0, trainable=False, name="prev_loss")
                self.delta_loss = tf.Variable(0.0, trainable=False, name="delta_loss")
            self.ema_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=scope.name)
            self.init_ema_vars = tf.variables_initializer(self.ema_vars)
#        if self.compute_fisher:
#            self._fishers, _, _, _ = compute_fishers(self.model)
#            #fishers = compute_fisher_information(model)
#            self.vars['fishers'] = dict(zip(weights, self._fishers))
#            #fishers, avg_fishers, update_fishers, zero_fishers = compute_fisher_information(model)

        def _var_update(vars, update_fn):
            updates = []
            for w in weights:
                updates.append(tf.assign(vars[w], update_fn(self.vars, w, vars[w])))
            return tf.group(*updates)

        def _compute_vars_update_op(updates):
            # Force task updates to happen sequentially
            update_op = tf.no_op()
            for name, update_fn in updates.items():
                with tf.control_dependencies([update_op]):
                    update_op = _var_update(self.vars[name], update_fn)
            return update_op

        self._vars_step_update_op = _compute_vars_update_op(self.step_updates)
        self._vars_task_update_op = _compute_vars_update_op(self.task_updates)
        self._vars_init_update_op = _compute_vars_update_op(self.init_updates)

        # Create task-relevant update ops
        reset_ops = []
        update_ops = []
        for name, metric_fn in self.task_metrics.items():
            metric = metric_fn(self)
            for w in weights:
                reset_ops.append(tf.assign(self.vars[name][w], 0*self.vars[name][w]))
                update_ops.append(tf.assign_add(self.vars[name][w], metric[w]))
        self._reset_task_metrics_op = tf.group(*reset_ops)
        self._update_task_metrics_op = tf.group(*update_ops)

        # Each step we update the weights using the optimizer as well as the step-specific variables
        self.step_op = tf.group(self._weight_update_op, self._vars_step_update_op)
        self.updates.append(self.step_op)
        # After each task, run task-specific variable updates
        self.task_op = self._vars_task_update_op
        self.init_op = self._vars_init_update_op

        if self.compute_average_weights:
            self.updates.append(self.maintain_weight_averages_op)

        if self.compute_average_loss:
            self.update_loss_op = tf.assign(self.prev_loss, self.ema_loss)
            bupdates = self.updates
            with tf.control_dependencies(bupdates + [self.update_loss_op]):
                self.updates = [tf.group(*[self.maintain_averages_op])]
            self.delta_loss = self.prev_loss - self.ema_loss

        return self.updates#[self._base_updates

    def init_task_vars(self):
        K.get_session().run([self.init_op])

    def init_acc_vars(self):
        K.get_session().run(self.init_ema_vars)

    def init_loss(self, X, y, batch_size):
        pass
        #sess = K.get_session()
        #xi, yi, sample_weights = self.model.model._standardize_user_data(X[:batch_size], y[:batch_size], batch_size=batch_size)
        #sess.run(tf.assign(self.prev_loss, self.initial_loss), {self.model.input:xi[0], self.model.model.targets[0]:yi[0], self.model.model.sample_weights[0]:sample_weights[0], K.learning_phase():1})

    def update_task_vars(self):
        K.get_session().run(self.task_op)

    def update_task_metrics(self, X, y, batch_size):
        # Reset metric accumulators
        n_batch = len(X) // batch_size

        sess = K.get_session()
        sess.run(self._reset_task_metrics_op)
        for i in range(n_batch):
            xi, yi, sample_weights = self.model.model._standardize_user_data(X[i * batch_size:(i+1) * batch_size], y[i*batch_size:(i+1)*batch_size], batch_size=batch_size)
            sess.run(self._update_task_metrics_op, {self.model.input:xi[0], self.model.model.targets[0]:yi[0], self.model.model.sample_weights[0]:sample_weights[0]})


    def reset_optimizer(self):
        """Reset the optimizer variables"""
        K.get_session().run(self.init_opt_vars)

    def get_config(self):
        raise ValueError("Write the get_config bro")

    def get_numvals_list(self, key='omega'):
        """ Returns list of numerical values such as for instance omegas in reproducible order """
        variables = self.vars[key]
        numvals = []
        for p in self.weights:
            numval = K.get_value(tf.reshape(variables[p],(-1,)))
            numvals.append(numval)
        return numvals

    def get_numvals(self, key='omega'):
        """ Returns concatenated list of numerical values such as for instance omegas in reproducible order """
        conc = np.concatenate(self.get_numvals_list(key))
        return conc

    def get_state(self):
        state = []
        vs = self.vars
        for key in vs.keys():
            if key=='oopt': continue
            v = vs[key]
            for p in v.values():
                state.append(K.get_value(p)) # FIXME WhyTF does this not work?
        return state

    def set_state(self, state):
        c = 0
        vs = self.vars
        for key in vs.keys():
            if key=='oopt': continue
            v = vs[key]
            for p in v.values():
                K.set_value(p,state[c])
                c += 1
