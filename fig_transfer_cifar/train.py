#!/usr/bin/python3
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

import sys, os
sys.path.extend([os.path.expanduser('..')])

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.cm as cmx

import seaborn as sns
sns.set_style("white")

from tqdm import trange, tqdm

import tensorflow as tf

from pathint import protocols
from pathint.optimizers import KOOptimizer
from keras.optimizers import Adam, RMSprop, SGD
from keras.callbacks import Callback
import keras.backend as K
import keras.activations as activations
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D

from pathint import utils
from pathint.keras_utils import LossHistory

# ## Parameters

# Data params

# Network params
input_shape = (3,32,32)

# size of pooling area for max pooling
pool_size = (2, 2)

# convolution kernel size
kernel_size = (3, 3)

# Optimization parameters
batch_size = 256
epochs_per_task = 60
learning_rate = 1e-3
nstats = 1 # repeats of experiment to compute stdev


add_evals=False # Saves evals accross runs
cvals = ['scratch', 0, 0.01, 0.05, 0.1, 0.2]
cvals = ['scratch', 0, 0.01, 0.1]
print("cvals %s"%cvals)


debug=False
if debug:
    cvals = [0.1] 
    epochs_per_task = 1 
    nstats = 1

# Reset optimizer after each age
reset_optimizer = True


# ## Construct datasets
n_tasks = 6
output_dim = 10*n_tasks
nb_classes = output_dim
# task_labels = [ range(i*10,(i+1)*10) for i in range(n_tasks) ]

task_labels, training_datasets = utils.construct_transfer_cifar10_cifar100(n_tasks, split='train')
_, validation_datasets = utils.construct_transfer_cifar10_cifar100(n_tasks, split='test')
print(task_labels)

# training_datasets = utils.construct_split_cifar10(task_labels, split='train')
# validation_datasets = utils.construct_split_cifar10(task_labels, split='test')

# ## Construct network, loss, and updates
tf.reset_default_graph()

config = tf.ConfigProto()
config.gpu_options.allow_growth=True
sess = tf.InteractiveSession(config=config)
sess.run(tf.global_variables_initializer())


# Instantiate masking functions
output_mask = tf.Variable(tf.zeros(output_dim), name="mask", trainable=False)

select = tf.select if hasattr(tf, 'select') else tf.where

def masked_softmax(logits):
    # logits are [batch_size, output_dim]
    x = select(tf.tile(tf.equal(output_mask[None, :], 1.0), [tf.shape(logits)[0], 1]), logits, -1e32 * tf.ones_like(logits))
    return activations.softmax(x)

def set_active_outputs(labels):
    new_mask = np.zeros(output_dim)
    for l in labels:
        new_mask[l] = 1.0
    sess.run(output_mask.assign(new_mask))
    # print("setting output mask")
    # print(sess.run(output_mask))
    
def masked_predict(model, data, targets):
    pred = model.predict(data)
    # print(pred)
    acc = np.argmax(pred,1)==np.argmax(targets,1)
    return acc.mean()

# Assemble the network model
model = Sequential()

model.add(Conv2D(32, (3, 3), padding='same',
                     input_shape=training_datasets[0][0].shape[1:]))
model.add(Activation('relu'))
model.add(Conv2D(32, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(64, (3, 3), padding='same'))
model.add(Activation('relu'))
model.add(Conv2D(64, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(512))
model.add(Activation('relu'))
model.add(Dropout(0.5))
# model.add(Dense(nb_classes))
model.add(Dense(nb_classes, kernel_initializer='zero', activation=masked_softmax))


# Define our training protocol
protocol_name, protocol = protocols.PATH_INT_PROTOCOL(omega_decay='sum', xi=1e-3 )
opt = Adam(lr=learning_rate, beta_1=0.9, beta_2=0.999)
# opt = RMSprop(lr=1e-3) 
# opt = SGD(1e-3)
oopt = KOOptimizer(opt, model=model, **protocol)
model.compile(loss='categorical_crossentropy', optimizer=oopt, metrics=['accuracy'])
model.model._make_train_function()

history = LossHistory()
callbacks = [history]
datafile_name = "split_cifar10_data_%s_lr%.2e_ep%i.pkl.gz"%(protocol_name, learning_rate, epochs_per_task)



def run_fits(cvals, training_data, valid_data, nstats=1):
    acc_mean = dict()
    acc_std = dict()
    for cidx, cval_ in enumerate(cvals):
        runs = []
        for runid in range(nstats):
            evals = []
            sess.run(tf.global_variables_initializer())
            # model.set_weights(saved_weights)
            cstuffs = []
            if cval_=='scratch':
                print("Scratch mode -- inits net before each age")
                cval = 0 
            else:
                print("setting cval")
                cval = cval_
            oopt.set_strength(cval)
            oopt.init_task_vars()
            print("cval is %f"%sess.run(oopt.lam))
            for age, tidx in enumerate(range(n_tasks)):
                if cval_=='scratch':
                    sess.run(tf.global_variables_initializer())
                    oopt.reset_optimizer()
                print("Age %i, cval is=%f"%(age,cval))
                set_active_outputs(task_labels[age])
                stuffs = model.fit(training_data[tidx][0], training_data[tidx][1], batch_size, epochs_per_task, callbacks=callbacks, verbose=0)
                oopt.update_task_metrics(training_data[tidx][0], training_data[tidx][1], batch_size)
                oopt.update_task_vars()
                ftask = []
                for j in range(n_tasks):
                    set_active_outputs(task_labels[j])
                    train_err = masked_predict(model, training_data[j][0], training_data[j][1])
                    valid_err = masked_predict(model, valid_data[j][0], valid_data[j][1])
                    ftask.append( (np.mean(valid_err), np.mean(train_err)) )
                evals.append(ftask)
                cstuffs.append(stuffs)

                # Re-initialize optimizater variables
                if reset_optimizer:
                    oopt.reset_optimizer()

            evals = np.array(evals)
            runs.append(evals)
        
        runs = np.array(runs)
        acc_mean[cval_] = runs.mean(0)
        acc_std[cval_] = runs.std(0)
    return dict(mean=acc_mean, std=acc_std)


# Run the sim
data = run_fits(cvals, training_datasets, validation_datasets, nstats=nstats)


# data = dict(mean={0.1:0.0}, std={0.1:0.0})
# print(data)
if add_evals:
    old_data = utils.load_zipped_pickle(datafile_name)
    # returns empty dict if file not found
    for k in old_data.keys():
        for l in old_data[k].keys():
            data[k][l] = old_data[k][l]

# Save the data 
utils.save_zipped_pickle(data, datafile_name)

# To overwrite the data in the file uncomment this
# all_evals = dict() # uncomment to delete on disk
# utils.save_zipped_pickle(data, datafile_name)
