# coding:utf-8
# models

import tensorflow as tf
from tensorflow.contrib import rnn
import configs as config


# Network parameters 
n_steps = config.FLAGS.word_size # total number of words in a training description
n_layers = 1  # number of the LSTM layers
keep_prob = 1.0
n_hidden = config.FLAGS.n_hidden # neurons in the hidden layer
batch_size = config.FLAGS.batch_size

# MLP for tag representations learning
# input:
#    x: [batch, height, width]   / [batch, step, embedding_size]
# output:
#    output: [batch, height, hidden_size]  / [batch, step, hidden_size]

def MLP(x, weights, biases):
    
    layer_addition = tf. matmul(x,  weights['in']) + biases['in']
    layer_activation = tf.nn.tanh(layer_addition)
    hidden_drop = tf.nn.dropout(layer_activation, keep_prob)
    output = tf.matmul(hidden_drop, weights['out']) + biases['out']
    return output

# x=[batch_size, n_steps, n_hidden]
def LSTM(x, weights, biases):
    x = tf.unstack(x, n_steps, 1)
#     x = tf. matmul(x,  weights['in']) + biases['in']
    lstm_cell = rnn.BasicLSTMCell(n_hidden, forget_bias=1.0)
    lstm_cell = rnn.DropoutWrapper(cell=lstm_cell, input_keep_prob=1.0, output_keep_prob=keep_prob)
    
    outputs, states = rnn.static_rnn(lstm_cell, x, dtype=tf.float32)
    return tf.matmul(outputs[-1], weights['out']) + biases['out']

def BiLSTM(x, weights, biases):
    
    x = tf.unstack(x, n_steps, 1)
    lstm_fw_cell = rnn.BasicLSTMCell(n_hidden, forget_bias=1.0, state_is_tuple=True)
    lstm_bw_cell = rnn.BasicLSTMCell(n_hidden, forget_bias=1.0, state_is_tuple=True)
    
    lstm_fw_cell = rnn.DropoutWrapper(cell=lstm_fw_cell, input_keep_prob=1.0, output_keep_prob=keep_prob)
    lstm_bw_cell = rnn.DropoutWrapper(cell=lstm_bw_cell, input_keep_prob=1.0, output_keep_prob=keep_prob)
    
    outputs, states = tf.nn.bidirectional_dynamic_rnn(lstm_fw_cell, lstm_bw_cell, x)
    return tf.matmul(outputs[-1], weights['out']) + biases['out']

    
# BiLSTM for functional and contextual service representations learning
# input:
#    x: [batch, height, width]   / [batch, step, embedding_size]
#    hidden_size: lstm hidden units
# output:
#    output: [batch, height, 2*hidden_size]  / [batch, step, 2*hidden_size]

def BiLSTM_expand(x, feature_weights, weights, biases):
    
    x = tf.unstack(x, n_steps, 1)
#     x = tf. matmul(x,  weights['in']) + biases['in']
    
    # forward direction
    with tf.name_scope("forward"), tf.variable_scope("forward"):
        stacked_lstm_fw = []
        for _ in range(n_layers):
            fw_cell = rnn.BasicLSTMCell(n_hidden, forget_bias=1.0, state_is_tuple=True)
            lstm_fw_cell = rnn.DropoutWrapper(fw_cell,output_keep_prob=keep_prob)
            stacked_lstm_fw.append(lstm_fw_cell)
        lstm_fw_cell_m = rnn.MultiRNNCell(cells=stacked_lstm_fw, state_is_tuple=True)
        outputs_fw = list()
        state_fw = fw_cell.zero_state(batch_size, tf.float32)
        for timestep in range(n_steps):
            if timestep > 0:
                tf.get_variable_scope().reuse_variables()
            output_fw, state_fw = lstm_fw_cell_m(x[:, timestep, :], state_fw)
            # add the functional attention mechanism
            weight_wi = tf.sigmoid(tf.multiply(feature_weights, output_fw))
            output_fw = tf.multiply(output_fw, weight_wi)
            outputs_fw.append(output_fw)
#     lstm1, state_h, state_c = LSTM(1, return_sequences=True, return_state=True)(inputs1)    
    # backward direction
    with tf.name_scope("backward"), tf.variable_scope("backward"):
        stacked_lstm_bw = []
        for _ in range(n_layers):
            bw_cell = rnn.BasicLSTMCell(n_hidden, forget_bias=1.0, state_is_tuple=True)
            lstm_bw_cell = rnn.DropoutWrapper(bw_cell,output_keep_prob=keep_prob)
            stacked_lstm_bw.append(lstm_bw_cell)
        lstm_bw_cell_m = rnn.MultiRNNCell(cells=stacked_lstm_bw, state_is_tuple=True)
        outputs_bw = list()
        state_bw = bw_cell.zero_state(batch_size, tf.float32)  
        x = tf.reverse(x, [1])
        for timestep in range(n_steps):
            if timestep > 0:
                tf.get_variable_scope.reuse_variables()
            output_bw, state_bw = lstm_bw_cell_m(x[:, timestep, :], state_bw)
            # add the functional attention mechanism
            weight_wi = tf.sigmoid(tf.multiply(feature_weights, output_bw))
            output_bw = tf.multiply(output_bw, weight_wi)
            outputs_bw.append(output_bw)
        
    outputs_bw = tf.reverse(outputs_bw, [0])
    output = tf.reduce_mean((([outputs_fw] + [outputs_bw]) / 2), 0)
    
    
#     # combine the forward and backward outputs
#     with tf.name_scope("bilstm"), tf.variable_scope("bilstm"):
#         output,_,_ = tf.contrib.rnn.static_bidirectional_rnn(lstm_fw_cell_m, 
#                                                              lstm_bw_cell_m, 
#                                                              x,
#                                                         dtype=tf.float32)
    return tf.matmul(output, weights['out']) + biases['out']

def TABiLSTM(x, weights_lstm, biases_lstm, weights_mlp, biases_mlp):
    
    x = tf.unstack(x, n_steps, 1)
    x = tf. matmul(x,  weights_lstm['in']) + biases_lstm['in']
    
    # forward direction
    with tf.name_scope("forward"), tf.variable_scope("forward"):
        stacked_lstm_fw = []
        for _ in range(n_layers):
            fw_cell = rnn.BasicLSTMCell(n_hidden, forget_bias=1.0, state_is_tuple=True)
            lstm_fw_cell = rnn.DropoutWrapper(fw_cell,output_keep_prob=keep_prob)
            stacked_lstm_fw.append(lstm_fw_cell)
        lstm_fw_cell_m = rnn.MultiRNNCell(cells=stacked_lstm_fw, state_is_tuple=True)
        outputs_fw = list()
        state_fw = fw_cell.zero_state(batch_size, tf.float32)
        for timestep in range(n_steps):
            if timestep > 0:
                tf.get_variable_scope().reuse_variables()
            output_fw, state_fw = lstm_fw_cell_m(x[:, timestep, :], state_fw)
            outputs_fw.append(output_fw)
        
    # backward direction
    with tf.name_scope("backward"), tf.variable_scope("backward"):
        stacked_lstm_bw = []
        for _ in range(n_layers):
            bw_cell = rnn.BasicLSTMCell(n_hidden, forget_bias=1.0, state_is_tuple=True)
            lstm_bw_cell = rnn.DropoutWrapper(bw_cell,output_keep_prob=keep_prob)
            stacked_lstm_bw.append(lstm_bw_cell)
        lstm_bw_cell_m = rnn.MultiRNNCell(cells=stacked_lstm_bw, state_is_tuple=True)
        outputs_bw = list()
        state_bw = bw_cell.zero_state(batch_size, tf.float32)  
        x = tf.reverse(x, [1])
        for timestep in range(n_steps):
            if timestep > 0:
                tf.get_variable_scope.reuse_variables()
            output_bw, state_bw = lstm_bw_cell_m(x[:, timestep, :], state_bw)
            outputs_bw.append(output_bw)
            
    output_mlp = MLP(x, weights_mlp, biases_mlp) # the functional context information (functional attention)
    
#     weight = tf.reshape(weight, [2,1])
    # reshape the output_mlp to [n_steps, 1]
    
    outputs_bw = tf.reverse(outputs_bw, [0])
    output_bidirection = ([outputs_fw] + [outputs_bw]) / 2 # combine the outputs from two opposite directions by element-wise mean
    
    # compute the functional attention score
#     output_mlp = tf.transpose(output_mlp, perm=[0, 2, 1])
    output_mlp = tf.reshape(output_mlp, [batch_size, n_steps, 1])
    attention_score = tf.matmul(output_bidirection, output_mlp)
    # attention mechanism
    output_bidirection = tf.multiply(output_bidirection, attention_score)
    # mean element-wise sum by utilizing both the word and tag representations
    output_word = tf.reduce_mean((output_bidirection), 0)
    output_tag = output_mlp
    output = (output_word + output_tag) / 2
    
    return tf.matmul(output, weights_lstm['out']) + biases_lstm['out']
    
# scenario_context=[batch_size, n_inputs]
def FCBiLSTM(x, scenario_context, weights_lstm, biases_lstm, weights_mlp, biases_mlp, waw, wam, wsw):
    x = tf.unstack(x, n_steps, 1)
    x = tf. matmul(x,  weights_lstm['in']) + biases_lstm['in']
    
    # forward direction
    with tf.name_scope("forward"), tf.variable_scope("forward"):
        stacked_lstm_fw = []
        for _ in range(n_layers):
            fw_cell = rnn.BasicLSTMCell(n_hidden, forget_bias=1.0, state_is_tuple=True)
            lstm_fw_cell = rnn.DropoutWrapper(fw_cell,output_keep_prob=keep_prob)
            stacked_lstm_fw.append(lstm_fw_cell)
        lstm_fw_cell_m = rnn.MultiRNNCell(cells=stacked_lstm_fw, state_is_tuple=True)
        outputs_fw = list()
        state_fw = fw_cell.zero_state(batch_size, tf.float32)
        for timestep in range(n_steps):
            if timestep > 0:
                tf.get_variable_scope().reuse_variables()
            output_fw, state_fw = lstm_fw_cell_m(x[:, timestep, :], state_fw)
            outputs_fw.append(output_fw)
        
    # backward direction
    with tf.name_scope("backward"), tf.variable_scope("backward"):
        stacked_lstm_bw = []
        for _ in range(n_layers):
            bw_cell = rnn.BasicLSTMCell(n_hidden, forget_bias=1.0, state_is_tuple=True)
            lstm_bw_cell = rnn.DropoutWrapper(bw_cell,output_keep_prob=keep_prob)
            stacked_lstm_bw.append(lstm_bw_cell)
        lstm_bw_cell_m = rnn.MultiRNNCell(cells=stacked_lstm_bw, state_is_tuple=True)
        outputs_bw = list()
        state_bw = bw_cell.zero_state(batch_size, tf.float32)
        x = tf.reverse(x, [1])
        for timestep in range(n_steps):
            if timestep > 0:
                tf.get_variable_scope.reuse_variables()
            output_bw, state_bw = lstm_bw_cell_m(x[:, timestep, :], state_bw)
            outputs_bw.append(output_bw)
            
    output_mlp = MLP(x, weights_mlp, biases_mlp) # the functional context information (functional attention)
    
#     weight = tf.reshape(weight, [2,1])
    # reshape the output_mlp to [n_steps, 1]
    
    outputs_bw = tf.reverse(outputs_bw, [0])
    output_bidirection = ([outputs_fw] + [outputs_bw]) / 2 # combine the outputs from two opposite directions by element-wise mean
    
    # compute the contextual attention
    ca = tf.multiply(output_bidirection, waw) + tf.multiply(scenario_context, wam)
    output_fw = tf.nn.tanh(tf.multiply(ca, wsw))
    
    # compute the functional attention score
#     output_mlp = tf.transpose(output_mlp, perm=[0, 2, 1])
    output_mlp = tf.reshape(output_mlp, [batch_size, n_steps, 1])
    attention_score = tf.matmul(output_bidirection, output_mlp)
    # attention mechanism
    output_bidirection = tf.multiply(output_bidirection, attention_score)
    # mean element-wise sum by utilizing both the word and tag representations
    output_word = tf.reduce_mean((output_bidirection), 0)
    output_tag = output_mlp
    output = (output_word + output_tag) / 2
    
    return tf.matmul(output, weights_lstm['out']) + biases_lstm['out']


    