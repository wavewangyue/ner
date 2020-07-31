import numpy as np
import tensorflow as tf
from bert import modeling as bert_modeling

class MyModel(object):
    
    def __init__(self, 
                 bert_config, 
                 vocab_size_bio, 
                 use_lstm, 
                 use_crf):
        
        self.inputs_seq = tf.placeholder(shape=[None, None], dtype=tf.int32, name="inputs_seq") # B * (S+2)
        self.inputs_mask = tf.placeholder(shape=[None, None], dtype=tf.int32, name="inputs_mask") # B * (S+2)
        self.inputs_segment = tf.placeholder(shape=[None, None], dtype=tf.int32, name="inputs_segment") # B * (S+2)
        self.outputs_seq = tf.placeholder(shape=[None, None], dtype=tf.int32, name='outputs_seq') # B * (S+2)

        inputs_seq_len = tf.reduce_sum(self.inputs_mask, axis=-1) # B
        
        bert_model = bert_modeling.BertModel(
            config=bert_config,
            is_training=True,
            input_ids=self.inputs_seq,
            input_mask=self.inputs_mask,
            token_type_ids=self.inputs_segment,
            use_one_hot_embeddings=False
        )
    
        bert_outputs = bert_model.get_sequence_output() # B * (S+2) * D
        
        if not use_lstm:
            hiddens = bert_outputs
        else:
            with tf.variable_scope('bilstm'):
                cell_fw = tf.nn.rnn_cell.LSTMCell(300)
                cell_bw = tf.nn.rnn_cell.LSTMCell(300)
                ((rnn_fw_outputs, rnn_bw_outputs), (rnn_fw_final_state, rnn_bw_final_state)) = tf.nn.bidirectional_dynamic_rnn(
                    cell_fw=cell_fw, 
                    cell_bw=cell_bw, 
                    inputs=bert_outputs, 
                    sequence_length=inputs_seq_len,
                    dtype=tf.float32
                )
                rnn_outputs = tf.add(rnn_fw_outputs, rnn_bw_outputs) # B * (S+2) * D
            hiddens = rnn_outputs
            
        with tf.variable_scope('projection'):
            logits_seq = tf.layers.dense(hiddens, vocab_size_bio) # B * (S+2) * V
            probs_seq = tf.nn.softmax(logits_seq)
            
            if not use_crf:
                preds_seq = tf.argmax(probs_seq, axis=-1, name="preds_seq") # B * S
            else:
                log_likelihood, transition_matrix = tf.contrib.crf.crf_log_likelihood(logits_seq, self.outputs_seq, inputs_seq_len)
                preds_seq, crf_scores = tf.contrib.crf.crf_decode(logits_seq, transition_matrix, inputs_seq_len)
            
        self.outputs = preds_seq
        
        with tf.variable_scope('loss'):
            if not use_crf: 
                loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits_seq, labels=self.outputs_seq) # B * S
                masks = tf.sequence_mask(inputs_seq_len, dtype=tf.float32) # B * S
                loss = tf.reduce_sum(loss * masks, axis=-1) / tf.cast(inputs_seq_len, tf.float32) # B
            else:
                loss = -log_likelihood / tf.cast(inputs_seq_len, tf.float32) # B
                
        self.loss = tf.reduce_mean(loss)
        
        with tf.variable_scope('opt'):
            params_of_bert = []
            params_of_other = []
            for var in tf.trainable_variables():
                vname = var.name
                if vname.startswith("bert"):
                    params_of_bert.append(var)
                else:
                    params_of_other.append(var)
            opt1 = tf.train.AdamOptimizer(1e-4)
            opt2 = tf.train.AdamOptimizer(1e-3)
            gradients_bert = tf.gradients(loss, params_of_bert)
            gradients_other = tf.gradients(loss, params_of_other)
            gradients_bert_clipped, norm_bert = tf.clip_by_global_norm(gradients_bert, 5.0)
            gradients_other_clipped, norm_other = tf.clip_by_global_norm(gradients_other, 5.0)
            train_op_bert = opt1.apply_gradients(zip(gradients_bert_clipped, params_of_bert))
            train_op_other = opt2.apply_gradients(zip(gradients_other_clipped, params_of_other))
        
        self.train_op = (train_op_bert, train_op_other)

        
        
