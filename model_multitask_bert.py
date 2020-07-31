import numpy as np
import tensorflow as tf
from bert import modeling as bert_modeling

class MyModel(object):
    
    def __init__(self, 
                 bert_config,
                 vocab_size_bio,
                 vocab_size_attr,
                 O_tag_index,
                 use_crf):
        
        self.inputs_seq = tf.placeholder(shape=[None, None], dtype=tf.int32, name="inputs_seq") # B * (S+2)
        self.inputs_mask = tf.placeholder(shape=[None, None], dtype=tf.int32, name="inputs_mask") # B * (S+2)
        self.inputs_segment = tf.placeholder(shape=[None, None], dtype=tf.int32, name="inputs_segment") # B * (S+2)
        self.outputs_seq_bio = tf.placeholder(tf.int32, [None, None], name='outputs_seq_bio') # B * (S+2)
        self.outputs_seq_attr = tf.placeholder(tf.int32, [None, None], name='outputs_seq_attr') # B * (S+2)

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
        
        with tf.variable_scope('bio_projection'):
            logits_bio = tf.layers.dense(bert_outputs, vocab_size_bio) # B * S * V
            probs_bio = tf.nn.softmax(logits_bio, axis=-1)
            
            if not use_crf:
                preds_bio = tf.argmax(probs_bio, axis=-1, name="preds_bio") # B * S
            else:
                log_likelihood, transition_matrix = tf.contrib.crf.crf_log_likelihood(logits_bio, 
                                                                                      self.outputs_seq_bio, 
                                                                                      inputs_seq_len)
                preds_bio, crf_scores = tf.contrib.crf.crf_decode(logits_bio, transition_matrix, inputs_seq_len)  

        with tf.variable_scope('attr_projection'):
            logits_attr = tf.layers.dense(bert_outputs, vocab_size_attr) # B * S * V
            probs_attr = tf.nn.softmax(logits_attr, axis=-1)
            preds_attr = tf.argmax(probs_attr, axis=-1, name="preds_attr") # B * S
        
        self.outputs = (preds_bio, preds_attr)
                
        with tf.variable_scope('loss'):
            if not use_crf:
                loss_bio = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits_bio, labels=self.outputs_seq_bio) # B * S
                masks_bio = tf.sequence_mask(inputs_seq_len, dtype=tf.float32) # B * S
                loss_bio = tf.reduce_sum(loss_bio * masks_bio, axis=-1) / tf.cast(inputs_seq_len, tf.float32) # B
            else:
                loss_bio = -log_likelihood / tf.cast(inputs_seq_len, tf.float32)
    
            loss_attr = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits_attr, labels=self.outputs_seq_attr) # B * S
            masks_attr = tf.cast(tf.not_equal(preds_bio, O_tag_index), tf.float32) # B * S
            loss_attr = tf.reduce_sum(loss_attr * masks_attr, axis=-1) / (tf.reduce_sum(masks_attr, axis=-1) + 1e-5) # B
            
            loss = loss_bio + loss_attr # B
        
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
        
