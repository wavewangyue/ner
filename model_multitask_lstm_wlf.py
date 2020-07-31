import tensorflow as tf
import random
import numpy as np

class MyModel(object):
    
    def __init__(self, 
                 embedding_dim, 
                 hidden_dim,
                 vocab_size_char,
                 vocab_size_word, 
                 vocab_size_bio, 
                 vocab_size_attr,
                 O_tag_index,
                 use_crf):
        
        self.inputs_seq_char = tf.placeholder(tf.int32, [None, None], name="inputs_seq_char")
        self.inputs_seq_word = tf.placeholder(tf.int32, [None, None], name="inputs_seq_word")
        self.inputs_seq_len = tf.placeholder(tf.int32, [None], name="inputs_seq_len")
        self.outputs_seq_bio = tf.placeholder(tf.int32, [None, None], name='outputs_seq_bio')
        self.outputs_seq_attr = tf.placeholder(tf.int32, [None, None], name='outputs_seq_attr')
        
        with tf.variable_scope('embedding_layer'):
            embedding_matrix_char = tf.get_variable("embedding_matrix_char", [vocab_size_char, embedding_dim], dtype=tf.float32)
            embedding_matrix_word = tf.get_variable("embedding_matrix_word", [vocab_size_word, embedding_dim], dtype=tf.float32)
            embedded_char = tf.nn.embedding_lookup(embedding_matrix_char, self.inputs_seq_char) # B * S * D
            embedded_word = tf.nn.embedding_lookup(embedding_matrix_word, self.inputs_seq_word) # B * S * D
            embedded = tf.concat([embedded_char, embedded_word], axis=2)
            self.embedding_matrix_word = embedding_matrix_word
        
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
            rnn_outputs = tf.add(rnn_fw_outputs, rnn_bw_outputs) # B * S * D
            
        with tf.variable_scope('bio_projection'):
            logits_bio = tf.layers.dense(rnn_outputs, vocab_size_bio) # B * S * V
            probs_bio = tf.nn.softmax(logits_bio, axis=-1)
            
            if not use_crf:
                preds_bio = tf.argmax(probs_bio, axis=-1, name="preds_bio") # B * S
            else:
                log_likelihood, transition_matrix = tf.contrib.crf.crf_log_likelihood(logits_bio, 
                                                                                      self.outputs_seq_bio, 
                                                                                      self.inputs_seq_len)
                preds_bio, crf_scores = tf.contrib.crf.crf_decode(logits_bio, transition_matrix, self.inputs_seq_len)    
        
        with tf.variable_scope('attr_projection'):
            logits_attr = tf.layers.dense(rnn_outputs, vocab_size_attr) # B * S * V
            probs_attr = tf.nn.softmax(logits_attr, axis=-1)
            preds_attr = tf.argmax(probs_attr, axis=-1, name="preds_attr") # B * S
        
        self.outputs = (preds_bio, preds_attr)
        
        with tf.variable_scope('loss'):
            if not use_crf:
                loss_bio = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits_bio, labels=self.outputs_seq_bio) # B * S
                masks_bio = tf.sequence_mask(self.inputs_seq_len, dtype=tf.float32) # B * S
                loss_bio = tf.reduce_sum(loss_bio * masks_bio, axis=-1) / tf.cast(self.inputs_seq_len, tf.float32) # B
            else:
                loss_bio = -log_likelihood / tf.cast(self.inputs_seq_len, tf.float32)
    
            loss_attr = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits_attr, labels=self.outputs_seq_attr) # B * S
            masks_attr = tf.cast(tf.not_equal(preds_bio, O_tag_index), tf.float32) # B * S
            loss_attr = tf.reduce_sum(loss_attr * masks_attr, axis=-1) / (tf.reduce_sum(masks_attr, axis=-1) + 1e-5) # B
            
            loss = loss_bio + loss_attr # B
        
        self.loss = tf.reduce_mean(loss)
            
        with tf.variable_scope('opt'):
            self.train_op = tf.train.AdamOptimizer().minimize(loss)


    
