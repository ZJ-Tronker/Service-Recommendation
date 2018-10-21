# coding:utf-8
# The FCLSTM model considering both the functional and contextual attention mechanisms

import tensorflow as tf
import numpy as np
from tensorflow.contrib import rnn
import configs as config
import os
import compnetwork as models
import data_loader as data_load
import random

class FCLSTM_model:
    
    def __init__(self, training_data, training_data_tags, n_step, n_input, n_output, n_hidden, batch_size, margin, epoches, n_step_tag, 
                 learning_rate=0.05, optimizer='SGD'):
        '''
        n_step: step size of the LSTM model
        n_input: the input embedding size of LSTM
        n_output: the output embedding size of LSTM
        n_hidden: number of the hidden units
        margin: margin of the hinge loss function
        learning_rate: learning rate of the SGD optimizer
        '''
        self.n_step = n_step
        self.n_input = n_input
        self.n_output = n_output
        self.n_hidden = n_hidden
        self.margin = margin
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.optimizer = optimizer
        self.epoches = epoches
        self.training_data = training_data
        self.n_step_tag = n_step_tag
        self.training_data_tags = training_data_tags
    
    def train(self):
        graph = tf.Graph()
        
        with graph.as_default():
            
            # input descriptions
            input_mashup_x = tf.placeholder(tf.float32, shape=[None, self.n_step, self.n_input])
            input_posservice_x = tf.placeholder(tf.float32, shape=[None, self.n_step, self.n_input])
            input_negservice_x = tf.placeholder(tf.float32, shape=[None, self.n_step, self.n_input])
            # input tags
            input_mashup_tag_x = tf.placeholder(tf.float32, shape=[None, self.n_step_tag, self.n_input])
            input_posservice_tag_x = tf.placeholder(tf.float32, shape=[None, self.n_step_tag, self.n_input])
            input_negservice_tag_x = tf.placeholder(tf.float32, shape=[None, self.n_step_tag, self.n_input])
            
            # weight and biases
            weights = {
                'in': tf.Variable(tf.random_normal([self.n_input, self.n_hidden])),
                'out': tf.Variable(tf.random_normal([self.n_hidden, self.n_output]))
                }
            biases = {
                'in': tf.Variable(tf.random_normal([self.n_hidden])),
                'out': tf.Variable(tf.random_normal([self.n_output]))
                }
            weights_mlp = {
                'in': tf.Variable(tf.random_normal([self.n_input, self.n_hidden])),
                'out': tf.Variable(tf.random_normal([self.n_hidden, self.n_output]))
                }
            biases_mlp = {
                'in': tf.Variable(tf.random_normal([self.n_hidden])),
                'out': tf.Variable(tf.random_normal([self.n_output]))
                }
            
#             weights = tf.Variable(tf.random_normal([self.n_hidden, self.n_output]))
#             biases = tf.Variable(tf.random_normal([self.n_output]))                
            
            weight_matrix = tf.Variable(tf.random_normal([self.n_output]))
            weights_aw = tf.Variable(tf.random_normal([self.n_step, 1]))
            weights_sw = tf.Variable(tf.random_normal([self.n_step, 1]))
            weights_am = tf.Variable(tf.random_normal([self.n_step, 1, 1]))
            
            with tf.variable_scope('MLP', reuse=tf.AUTO_REUSE):
                mlp_mashup_tag = tf.reduce_mean(models.MLP(input_mashup_tag_x, weights_mlp, biases_mlp), 1)
                mlp_posservice_tag = tf.reduce_mean(models.MLP(input_posservice_tag_x, weights_mlp, biases_mlp), 1)
                mlp_negservice_tag = tf.reduce_mean(models.MLP(input_negservice_tag_x, weights_mlp, biases_mlp), 1)
                
                # calculate weight_w_i
                
            # define the FCLSTM model for service and Mashup descriptions
            with tf.variable_scope('bilstm', reuse=tf.AUTO_REUSE):
                output_mashup = models.BiLSTM_expand(input_mashup_x, tf.multiply(mlp_mashup_tag, weight_matrix), weights, biases)
                # add the tag information 
                output_mashup = output_mashup + mlp_mashup_tag
                
                # add the contextual attention
                first_term_pos = tf.multiply(input_posservice_x, weights_aw)
                first_term_neg = tf.multiply(input_negservice_x, weights_aw)
                
                second_term = tf.multiply(output_mashup, weights_am)
                second_term = tf.transpose(second_term, [1,0,2])
                
                # calculate ca(w_i)
                input_posservice_x = tf.tanh(first_term_pos + second_term)
                input_negservice_x = tf.tanh(first_term_neg + second_term)
                
                # calculate the new h_t_w_i
                temp_exp = tf.exp(tf.multiply(input_posservice_x, weights_sw))
                
                input_posservice_x = tf.multiply(input_posservice_x, temp_exp)
                input_negservice_x = tf.multiply(input_negservice_x, temp_exp)
                
                output_posservice = models.BiLSTM_expand(input_posservice_x, tf.multiply(mlp_posservice_tag, weight_matrix), weights, biases)
                output_negservice = models.BiLSTM_expand(input_negservice_x, tf.multiply(mlp_negservice_tag, weight_matrix), weights, biases)
                
                # add the tag information 
                output_posservice = output_posservice + mlp_posservice_tag
                output_negservice = output_negservice + mlp_negservice_tag
                
            # calculate the cosine similarity
            MD_len = tf.sqrt(tf.reduce_sum(tf.multiply(output_mashup, output_mashup), 1))
            PSD_len = tf.sqrt(tf.reduce_sum(tf.multiply(output_posservice, output_posservice), 1))
            NSD_len = tf.sqrt(tf.reduce_sum(tf.multiply(output_negservice, output_negservice), 1))
            MPSD_dist = tf.sqrt(tf.reduce_sum(tf.multiply(output_mashup, output_posservice), 1))
            MNSD_dist = tf.sqrt(tf.reduce_sum(tf.multiply(output_mashup, output_negservice), 1))
            
            with tf.name_scope("score"):
                MPSD_score = tf.div(MPSD_dist, tf.multiply(MD_len, PSD_len))
                MNSD_score = tf.div(MNSD_dist, tf.multiply(MD_len, NSD_len))

            # calculate the hinge loss
            zero = tf.fill(tf.shape(MPSD_score), 0.0)
            margin = tf.fill(tf.shape(MPSD_score), self.margin)
            with tf.name_scope("loss"):
                global_loss = tf.reduce_mean(tf.maximum(zero, tf.subtract(margin, tf.subtract(MPSD_score, MNSD_score))))
            
            # Optimizer.
            if self.optimizer == 'Adagrad':
#                 global_step = tf.Variable(1, name="global_step", trainable=False)
#                 optimizer1 = tf.train.AdamOptimizer(self.learning_rate)
#                 grads_and_vars = optimizer1.compute_gradients(global_loss)
#                 grads, _ = list(zip(*grads_and_vars))
#                 tf.summary.scalar("gradient_norm", tf.global_norm(grads))
#                 optimizer = optimizer1.apply_gradients(grads_and_vars=grads_and_vars, global_step=global_step,
#                                                           name="train_op")
                global_step = tf.Variable(1, name="global_step", trainable=False)
                optimizer = tf.train.AdagradOptimizer(self.learning_rate).minimize(global_loss, global_step=global_step)
#                 optimizer = tf.train.AdagradOptimizer(self.learning_rate).minimize(global_loss)
            elif self.optimizer == 'SGD':
                optimizer = tf.train.GradientDescentOptimizer(self.learning_rate).minimize(global_loss)

            # Add variable initializer
            init = tf.global_variables_initializer()
            config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
            with tf.Session(graph=graph,config=config) as sess:        
#             with tf.Session(graph=graph) as sess:        
                init.run()
                print("Initialized!")
#                 training_data = data_load.load_data() 
                training_data = list(self.training_data)
                training_data_tags = list(self.training_data_tags)

                mini_batch_size = self.batch_size
                print(len(training_data))
                training_size = len(training_data)
                
                mini_batches = [(training_data[k:k+mini_batch_size], training_data_tags[k:k+mini_batch_size]) for k in range(0, training_size, mini_batch_size)]
                sup_length = mini_batch_size - len(mini_batches[-1])
                mini_batches[-1] += random.sample(training_data, sup_length)
                for i in range(self.epoches):
                    training_losses = 0.0
                    for mini_batch in mini_batches:
                        desc_mini_batch = mini_batch[0]
                        tag_mini_batch = mini_batch[1]
                        
                        mashups, services_pos, services_neg = zip(*desc_mini_batch)
                        mashups, services_pos, services_neg = np.array(mashups), np.array(services_pos), np.array(services_neg)
                        mashup_tags, services_pos_tags, services_neg_tags = zip(*tag_mini_batch)
                        mashup_tags, services_pos_tags, services_neg_tags = np.array(mashup_tags), np.array(services_pos_tags), np.array(services_neg_tags)

                        # backpropagation for optimization
                        sess.run(optimizer, feed_dict={input_mashup_x:mashups, 
                                                       input_posservice_x:services_pos, input_negservice_x:services_neg,
                                                       input_mashup_tag_x: mashup_tags,
                                                       input_posservice_tag_x: services_pos_tags,
                                                       input_negservice_tag_x: services_neg_tags})
                        training_loss = sess.run(global_loss, feed_dict={input_mashup_x:mashups, 
                                                       input_posservice_x:services_pos, input_negservice_x:services_neg,
                                                       input_mashup_tag_x: mashup_tags,
                                                       input_posservice_tag_x: services_pos_tags,
                                                       input_negservice_tag_x: services_neg_tags})
                        training_losses += training_loss
                    average_traing_loss = training_losses / len(mini_batches)
                    print("Epoch: {}, Training data - loss= {:.6f}".format(i, average_traing_loss))

def main():
    
    training_data, training_data_tags = data_load.load_data_withtags()

    n_steps= config.FLAGS.word_size
    n_input = config.FLAGS.embedding_size
    n_output = config.FLAGS.n_output
    n_hidden = config.FLAGS.n_hidden
    batch_size = config.FLAGS.batch_size
    margin = config.FLAGS.margin
    learning_rate = config.FLAGS.learning_rate
    optimizer = config.FLAGS.optimizer
    epoches = config.FLAGS.epoches
    n_step_tag = config.FLAGS.tag_size
    
    lstm_model = FCLSTM_model(training_data, training_data_tags, n_steps, n_input, n_output, n_hidden, batch_size, 
                           margin, epoches, n_step_tag, learning_rate, optimizer)
    
    lstm_model.train()
    
if __name__ == '__main__':
    main()