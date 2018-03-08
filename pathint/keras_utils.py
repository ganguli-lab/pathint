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

# Keras-specific functions and utils
from keras.callbacks import Callback
from keras.models import Model
import keras.backend as K
from keras.layers import Dense
import tensorflow as tf

class LossHistory(Callback):
    def __init__(self, *args, **kwargs):
        super(LossHistory, self).__init__(*args, **kwargs)
        self.losses = []
        self.regs = []

    def on_batch_end(self, batch, logs={}):
        self.losses.append(logs.get('loss'))
        self.regs.append(K.get_session().run(self.model.optimizer.regularizer))


# Create a callback that tracks FisherInformation
from keras.models import Model
import keras.backend as K
def compute_fishers(model):
    # Check that model only contains Dense layers
    for l in model.layers:
        if not isinstance(l, Dense):
            raise ValueError("All layers of the model must be Dense, got %s"%l)
    # Create new model used to extract activations at each layer
    new_model = Model(inputs=model.input, outputs=[l.output for l in model.layers])
    acts = new_model(model.input)

    out = acts[-1]
    out_dim = out.get_shape().as_list()[-1]
    #assert len(model.weights) %2 == 0
    n_weights = len(model.weights) // 2


    def _get_fishers():
        fisher_weights = [0.0] * n_weights
        fisher_biases = [0.0] * n_weights
        for idx in range(out_dim):
            # Clips output 
            # https://github.com/fchollet/keras/blob/master/keras/backend/tensorflow_backend.py#L2743
            output = out[:, idx]
            epsilon = tf.convert_to_tensor(x)
            if epsilon.dtype != output.dtype.base_dtype:
                epsilon = tf.cast(epsilon, output.dtype.base_dtype)

            output = tf.clip_by_value(output, epsilon, 1. - epsilon)
            y = K.log(output)
            # From the post-nonlinearity outputs of each layer, we walk back up through the graph
            # to find the linear activation corresponding to h=XW + b.
            # Then we identify the weights W, biases b, and previous activation X.
            # 1. TensorFlow can compute dy/dh, giving us a [batch_size, n_neurons] matrix
            # 2. We manually comute dy/dW=dy/dh X and dy/db=dy/dh
            # 3. We sum the squared Jacobians

            # Identify pre-nonlinearity activation corresponding to h=XW+b
            def _walk_up_until_add(x):
                if x.op.type == 'BiasAdd':
                    return x
                elif x.op.type == 'Select':
                    return x.op.inputs[1]
                else:
                    return _walk_up_until_add(x.op.inputs[0])

            linear = [_walk_up_until_add(a) for a in acts]
            # Identify previous activation, X
            prev_acts = [l.op.inputs[0].op.inputs[0] for l in linear]
            # Compute dy/dh
            dy_dlinear = [tf.gradients(y,l)[0] for l in linear]

            # Figure out which Jacobians correspond to which weights
            if idx == 0:
                val_to_var = {v.value():v for v in  model.weights}
                weight_vars = [val_to_var[l.op.inputs[0].op.inputs[1]] for l in linear]
                bias_vars = [val_to_var[l.op.inputs[1]] for l in linear]

            # Compute the sum of the Jacobian squared
            # Because each of the Jacobians are rank-1, we can compute this by first squaring and then summing:
            # \sum_i (u_i v_i^T)^2 = \sum_i (u_i^2 (v_i^2)^T)
            weights_sum_jacobian_squared = [tf.matmul(tf.transpose(a)**2, dh**2) for dh,a in zip(dy_dlinear, prev_acts)]
            bias_sum_jacobian_squared = [tf.reduce_sum(dh**2, 0) for dh in dy_dlinear]
            # Keep track of aggregate across outputs
            for jj in range(n_weights):
                fisher_weights[jj] += weights_sum_jacobian_squared[jj]
                fisher_biases[jj] += bias_sum_jacobian_squared[jj]
        var_to_fisher = dict(zip(weight_vars+bias_vars, fisher_weights+fisher_biases))
        return {w: var_to_fisher[w] for w in model.weights}

    fishers = _get_fishers()
    return fishers

    # Allocate space for accumulated Fisher
    avg_fishers = [K.zeros(w.get_shape().as_list()) for w in model.weights]
    # Create updates to reset avg fisher, update, etc.
    update_fishers = tf.group(*[tf.assign_add(avg_f, f) for avg_f, f in zip(avg_fishers, fishers)])
    zero_fishers = tf.group(*[tf.assign(avg_f, 0.0 * avg_f) for avg_f in avg_fishers])
    return fishers, avg_fishers, update_fishers, zero_fishers

