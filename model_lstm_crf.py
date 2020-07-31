import tensorflow as tf
import random
import numpy as np

class MyModel(object):
    
    def __init__(self, 
                 embedding_dim, 
                 hidden_dim, 
                 vocab_size_char, 
                 vocab_size_bio, 
                 use_crf):
        
        self.inputs_seq = tf.placeholder(tf.int32, [None, None], name="inputs_seq")
        self.inputs_seq_len = tf.placeholder(tf.int32, [None], name="inputs_seq_len")
        self.outputs_seq = tf.placeholder(tf.int32, [None, None], name='outputs_seq')
        
        with tf.variable_scope('embedding_layer'):
            embedding_matrix = tf.get_variable("embedding_matrix", [vocab_size_char, embedding_dim], dtype=tf.float32)
            embedded = tf.nn.embedding_lookup(embedding_matrix, self.inputs_seq)
        
        with tf.variable_scope('encoder'):
            cell_fw = tf.nn.rnn_cell.LSTMCell(hidden_dim)
            cell_bw = tf.nn.rnn_cell.LSTMCell(hidden_dim)
            ((rnn_fw_outputs, rnn_bw_outputs), (rnn_fw_final_state, rnn_bw_final_state)) = tf.nn.bidirectional_dynamic_rnn(
                cell_fw=cell_fw, 
                cell_bw=cell_bw, 
                inputs=embedded, 
                sequence_length=self.inputs_seq_len,
                dtype=tf.float32
            )
            rnn_outputs = tf.add(rnn_fw_outputs, rnn_bw_outputs) # B * S1 * D
        
        with tf.variable_scope('projection'):
            logits_seq = tf.layers.dense(rnn_outputs, vocab_size_bio) # B * S * V
            probs_seq = tf.nn.softmax(logits_seq)
            
            if not use_crf:
                preds_seq = tf.argmax(probs_seq, axis=-1, name="preds_seq") # B * S
            else:
                log_likelihood, transition_matrix = tf.contrib.crf.crf_log_likelihood(logits_seq, self.outputs_seq, self.inputs_seq_len)
                preds_seq, crf_scores = tf.contrib.crf.crf_decode(logits_seq, transition_matrix, self.inputs_seq_len)
            
        self.outputs = preds_seq
        
        with tf.variable_scope('loss'):
            if not use_crf: 
                loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits_seq, labels=self.outputs_seq) # B * S
                masks = tf.sequence_mask(self.inputs_seq_len, dtype=tf.float32) # B * S
                loss = tf.reduce_sum(loss * masks, axis=-1) / tf.cast(self.inputs_seq_len, tf.float32) # B
            else:
                loss = -log_likelihood / tf.cast(self.inputs_seq_len, tf.float32) # B
            
        self.loss = tf.reduce_mean(loss)
        
        with tf.variable_scope('opt'):
            self.train_op = tf.train.AdamOptimizer().minimize(loss)


    
