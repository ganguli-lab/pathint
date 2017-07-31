import tensorflow as tf

def quadratic_regularizer(weights, vars, norm=2):
    """Compute the regularization term.

    Args:
        weights: list of Variables
        _vars: dict from variable name to dictionary containing the variables.
            Each set of variables is stored as a dictionary mapping from weights to variables.
            For example, vars['grads'][w] would retreive the 'grads' variable for weight w
        norm: power for the norm of the (weights - consolidated weight)

    Returns:
        scalar Tensor regularization term
    """
    reg = 0.0
    for w in weights:
        reg += tf.reduce_sum(vars['omega'][w] * (w - vars['cweights'][w])**norm)
    return reg

def get_power_regularizer(power=2.0):
    """Power regularizers with different norms"""
    def _regularizer_fn(weights, vars):
        return quadratic_regularizer(weights, vars, norm=power)
    return _regularizer_fn
