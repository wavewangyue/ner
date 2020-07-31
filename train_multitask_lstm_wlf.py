import logging
import tensorflow as tf
import numpy as np
import os

from model_multitask_lstm_wlf import MyModel
from utils import DataProcessor_MTL_LSTM_WLF as DataProcessor
from utils import load_vocabulary
from utils import extract_kvpairs_in_bioes
from utils import cal_f1_score

# set logging
log_file_path = "./ckpt/run.log"
if os.path.exists(log_file_path): os.remove(log_file_path)
logger = logging.getLogger()
logger.setLevel(logging.INFO)
formatter = logging.Formatter("%(asctime)s | %(message)s", "%Y-%m-%d %H:%M:%S")
chlr = logging.StreamHandler()
chlr.setFormatter(formatter)
fhlr = logging.FileHandler(log_file_path)
fhlr.setFormatter(formatter)
logger.addHandler(chlr)
logger.addHandler(fhlr)

logger.info("loading vocab...")

w2i_char, i2w_char = load_vocabulary("./data/vocab_char.txt")
w2i_word, i2w_word = load_vocabulary("./data/vocab_word.txt")
w2i_bio, i2w_bio = load_vocabulary("./data/vocab_bio.txt")
w2i_attr, i2w_attr = load_vocabulary("./data/vocab_attr.txt")

logger.info("loading data...")

data_processor_train = DataProcessor(
    "./data/train/input.seq.char",
    "./data/train/input.seq.word",
    "./data/train/output.seq.bio",
    "./data/train/output.seq.attr",
    w2i_char,
    w2i_word,
    w2i_bio,
    w2i_attr,
    shuffling=True
)

data_processor_valid = DataProcessor(
    "./data/test/input.seq.char",
    "./data/test/input.seq.word",
    "./data/test/output.seq.bio",
    "./data/test/output.seq.attr",
    w2i_char,
    w2i_word,
    w2i_bio,
    w2i_attr,
    shuffling=True
)

logger.info("building model...")

model = MyModel(embedding_dim=300,
                hidden_dim=300,
                vocab_size_char=len(w2i_char),
                vocab_size_word=len(w2i_word),
                vocab_size_bio=len(w2i_bio),
                vocab_size_attr=len(w2i_attr),
                O_tag_index=w2i_bio["O"],
                use_crf=False)

logger.info("model params:")
params_num_all = 0
for variable in tf.trainable_variables():
    params_num = 1
    for dim in variable.shape:
        params_num *= dim
    params_num_all += params_num
    logger.info("\t {} {} {}".format(variable.name, variable.shape, params_num))
logger.info("all params num: " + str(params_num_all))
        
logger.info("start training...")

tf_config = tf.ConfigProto(allow_soft_placement=True)
tf_config.gpu_options.allow_growth = True

with tf.Session(config=tf_config) as sess:
    sess.run(tf.global_variables_initializer())
    saver = tf.train.Saver(max_to_keep=50)
    
    epoches = 0
    losses = []
    batches = 0
    best_f1 = 0
    batch_size = 32

    while epoches < 20:
        (inputs_seq_char_batch, 
         inputs_seq_word_batch, 
         inputs_seq_len_batch,
         outputs_seq_bio_batch, 
         outputs_seq_attr_batch) = data_processor_train.get_batch(batch_size)
        
        feed_dict = {
            model.inputs_seq_char: inputs_seq_char_batch,
            model.inputs_seq_word: inputs_seq_word_batch,
            model.inputs_seq_len: inputs_seq_len_batch,
            model.outputs_seq_bio: outputs_seq_bio_batch,
            model.outputs_seq_attr: outputs_seq_attr_batch
        }
        
        if batches == 0: 
            logger.info("###### shape of a batch #######")
            logger.info("input_seq_char: " + str(inputs_seq_char_batch.shape))
            logger.info("input_seq_word: " + str(inputs_seq_word_batch.shape))
            logger.info("input_seq_len: " + str(inputs_seq_len_batch.shape))
            logger.info("output_seq_bio: " + str(outputs_seq_bio_batch.shape))
            logger.info("output_seq_attr: " + str(outputs_seq_attr_batch.shape))
            logger.info("###### preview a sample #######")
            logger.info("input_seq_char:" + " ".join([i2w_char[i] for i in inputs_seq_char_batch[0]]))
            logger.info("input_seq_word:" + " ".join([i2w_word[i] for i in inputs_seq_word_batch[0]]))
            logger.info("input_seq_len :" + str(inputs_seq_len_batch[0]))
            logger.info("output_seq_bio: " + " ".join([i2w_bio[i] for i in outputs_seq_bio_batch[0]]))
            logger.info("output_seq_attr: " + " ".join([i2w_attr[i] for i in outputs_seq_attr_batch[0]]))
            logger.info("###############################")
        
        loss, _ = sess.run([model.loss, model.train_op], feed_dict)
        losses.append(loss)
        batches += 1
        
        if data_processor_train.end_flag:
            data_processor_train.refresh()
            epoches += 1

        def valid(data_processor, max_batches=None, batch_size=1024):
            preds_kvpair = []
            golds_kvpair = []
            batches_sample = 0
            
            while True:
                (inputs_seq_char_batch, 
                 inputs_seq_word_batch, 
                 inputs_seq_len_batch,
                 outputs_seq_bio_batch,
                 outputs_seq_attr_batch) = data_processor.get_batch(batch_size)

                feed_dict = {
                    model.inputs_seq_char: inputs_seq_char_batch,
                    model.inputs_seq_word: inputs_seq_word_batch,
                    model.inputs_seq_len: inputs_seq_len_batch,
                    model.outputs_seq_bio: outputs_seq_bio_batch,
                    model.outputs_seq_attr: outputs_seq_attr_batch
                }

                preds_seq_bio_batch, preds_seq_attr_batch = sess.run(model.outputs, feed_dict)
                
                for pred_seq_bio, gold_seq_bio, pred_seq_attr, gold_seq_attr, input_seq, l in zip(preds_seq_bio_batch, 
                                                                                                  outputs_seq_bio_batch, 
                                                                                                  preds_seq_attr_batch, 
                                                                                                  outputs_seq_attr_batch,
                                                                                                  inputs_seq_char_batch, 
                                                                                                  inputs_seq_len_batch):
                    pred_seq_bio = [i2w_bio[i] for i in pred_seq_bio[:l]]
                    gold_seq_bio = [i2w_bio[i] for i in gold_seq_bio[:l]]
                    char_seq = [i2w_char[i] for i in input_seq[:l]]
                    pred_seq_attr = [i2w_attr[i] for i in pred_seq_attr[:l]]
                    gold_seq_attr = [i2w_attr[i] for i in gold_seq_attr[:l]]
                    pred_kvpair = extract_kvpairs_in_bioes(pred_seq_bio, char_seq, pred_seq_attr)
                    gold_kvpair = extract_kvpairs_in_bioes(gold_seq_bio, char_seq, gold_seq_attr)
                    
                    preds_kvpair.append(pred_kvpair)
                    golds_kvpair.append(gold_kvpair)
                    
                if data_processor.end_flag:
                    data_processor.refresh()
                    break
                
                batches_sample += 1
                if (max_batches is not None) and (batches_sample >= max_batches):
                    break
            
            p, r, f1 = cal_f1_score(preds_kvpair, golds_kvpair)
            
            logger.info("Valid Samples: {}".format(len(preds_kvpair)))
            logger.info("Valid P/R/F1: {} / {} / {}".format(round(p*100, 2), round(r*100, 2), round(f1*100, 2)))

            return (p, r, f1)
            
        if batches % 100 == 0:
            logger.info("")
            logger.info("Epoches: {}".format(epoches))
            logger.info("Batches: {}".format(batches))
            logger.info("Loss: {}".format(sum(losses) / len(losses)))
            losses = []

            ckpt_save_path = "./ckpt/model.ckpt.batch{}".format(batches)
            logger.info("Path of ckpt: {}".format(ckpt_save_path))
            saver.save(sess, ckpt_save_path)
            
            p, r, f1 = valid(data_processor_valid, max_batches=10)
            if f1 > best_f1:
                best_f1 = f1
                logger.info("############# best performance now here ###############")
            
            