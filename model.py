# Model for training a LSTM to predict student test scores
# See Piesch 2015 https://arxiv.org/abs/1506.05908

import numpy as np
import pandas as pd
import tensorflow as tf


class Model:
    def __init__(   
            self,
            batch_size,
            num_probs,
            embedding_size=200,
            num_hid=200,
            initial_learning_rate=0.001,
            final_learning_rate=0.00001,
            keep_prob=0.5,
            epsilon=0.001):
        """Create an LSTM for test score prediction
        Args:
            batch_size: minibatch size
            num_probs: number of unique problems in the dataset
            embedding_size: size of the embedding lookup layer
            num_hid: number of units in the single hidden layer
            initial_learning_rate: learning rate of ADAM optmizer at first step of training
            epsilon: epsilon parameter of ADAM optimizer
            final_learning_rate: learning rate after 3000 training steps (linear decay)
            keep_prob: keep probability for dropout layer
        Returns:
            tuple: (optimizer, training_loss, validation_prediction)
        """

        # Inputs
        Xs = self._Xs = tf.placeholder(tf.int32, shape=[batch_size, None])
        Ys = self._Ys = tf.placeholder(tf.float32, shape=[batch_size, None, num_probs])
        targets = self._targets = tf.placeholder(tf.float32, shape=[batch_size, None])
        sequence_length = self._seqlen = tf.placeholder(tf.int32, shape=[batch_size])

        # Global parameters
        global_step = tf.Variable(0, trainable=False)
        learning_rate = tf.train.polynomial_decay(initial_learning_rate, global_step,5000,final_learning_rate)

        # LSTM parameters
        w = tf.Variable(tf.truncated_normal([num_hid, num_probs],stddev=1.0/np.sqrt(num_probs)))
        b = tf.Variable(tf.truncated_normal([num_probs],stddev=1.0/np.sqrt(num_probs)))
        embeddings = tf.Variable(tf.random_uniform([2*num_probs+2, embedding_size], -1.0, 1.0))
        cell = tf.nn.rnn_cell.BasicLSTMCell(num_hid)
        initial_state = cell.zero_state(batch_size,tf.float32)

        # LSTM Training ops
        inputsX = tf.nn.embedding_lookup(embeddings,Xs)
        outputs, state = tf.nn.dynamic_rnn(cell,inputsX,sequence_length,initial_state=initial_state)
        if keep_prob != 1:
            outputs = tf.nn.dropout(outputs, keep_prob)
        outputs_flat = tf.reshape(outputs,shape=[-1,num_hid])
        logits = tf.reshape(tf.nn.xw_plus_b(outputs_flat, w, b),shape=[batch_size,-1,num_probs])
        pred = tf.reduce_max(logits * Ys, axis=2)
        loss = tf.nn.sigmoid_cross_entropy_with_logits(logits=pred, labels=targets)
        mask = tf.sign(tf.abs(pred))
        loss_masked = mask*loss
        loss_masked_by_s = tf.reduce_sum(loss_masked, axis=1)
        mean_loss = self._loss = tf.reduce_mean(loss_masked_by_s/tf.to_float(sequence_length))

        # Optimizer
        optimizer = self._train = tf.train.AdamOptimizer(learning_rate=learning_rate,
            epsilon=epsilon).minimize(mean_loss,global_step=global_step)
        
        saver = self._saver = tf.train.Saver()

        # LSTM Validation ops
        test_outputs, test_state = tf.nn.dynamic_rnn(cell,inputsX,sequence_length,initial_state)
        test_outputs_flat = tf.reshape(test_outputs, shape=[-1,num_hid])
        test_logits = tf.reshape(tf.nn.xw_plus_b(test_outputs_flat,w,b),shape=[batch_size,-1,num_probs])
        test_pred = self._pred = tf.sigmoid(tf.reduce_max(test_logits*Ys, axis=2))

    @property
    def Xs(self):
        return self._Xs

    @property
    def Ys(self):
        return self._Ys

    @property
    def targets(self):
        return self._targets

    @property
    def seq_len(self):
        return self._seqlen

    @property
    def loss(self):
        return self._loss

    @property
    def train_op(self):
        return self._train

    @property
    def predict(self):
        return self._pred

    @property
    def saver(self):
        return self._saver