import argparse

import model
from load import read_data_from_csv_file

import tensorflow as tf
import numpy as np
from sklearn.metrics import roc_auc_score

def one_hot(hot,size):
    v = np.zeros(size)
    v[hot] = 1.0
    return v

class BatchGenerator:
    """
    Holds test score data and serves minibatches for training
    """

    def __init__(self,data,batch_size,id2idx,max_len=200):
        """
        Args:
            data: score data in the format [[sequence_length, [(problem_id, success_or_fail)]]]
                    each sequence is a different test taker
                    list of test scores is in temporal order
            batch_size: minibatch size to serve
            max_len: the longest sequence of responses to keep
            id2idx: dictionary mapping problem ids to unique indices in [0,n_probs]
        """
        self.data = sorted(data, key = lambda x: len(x[1]))
        self.cursor = 0
        self.batch_size = batch_size
        self.max_len = max_len
        self.num_probs = len(id2idx)
        self.id2idx = id2idx
        self.n = len(data)
            
    def next_batch(self):
        tups = []
        n_samps = []
        for i in range(self.batch_size):
            tups.append(self.data[self.cursor][1][:self.max_len])            
            n_samps.append(len(tups[-1]))
            self.cursor = (self.cursor + 1) % len(self.data)
        mlen = max(n_samps)
        Xs = np.zeros((self.batch_size, mlen),dtype=np.int32)
        Ys = np.zeros((self.batch_size, mlen, self.num_probs),dtype=np.int32)
        targets = np.zeros((self.batch_size, mlen),dtype=np.int32)
        for i, tuplist in enumerate(tups):
            Xs[i] = np.pad([2 + self.id2idx[t[0]] + t[1]*self.num_probs for t in tuplist[:-1]],
                (1,mlen-len(tuplist)),'constant',constant_values=(1,0))
            Ys[i] = np.pad([one_hot(self.id2idx[t[0]],self.num_probs) for t in tuplist],
                ((0,mlen-len(tuplist)),(0,0)),'constant',constant_values=0)
            targets[i] = np.pad([t[1] for t in tuplist],(0,mlen-len(tuplist)),'constant',constant_values=0)
        return Xs, Ys, targets, n_samps

def run(session, train_batchgen, test_batchgen, train_steps):

    m = model.Model(train_batchgen.batch_size,train_batchgen.num_probs)
    with session.as_default() as sess:
        tf.global_variables_initializer().run()
        print("Initialized")
        average_loss = 0
        for step in range(train_steps):
            batch_Xs, batch_Ys, batch_labels, batch_sequence_lengths = train_batches.next_batch()
            feed_dict = {m.Xs : batch_Xs, m.Ys : batch_Ys, 
                         m.seq_len : batch_sequence_lengths, m.targets : batch_labels}
            _, l = session.run([m.train_op,m.loss], feed_dict=feed_dict)
            average_loss += l
            if step % 100 == 0:
                average_loss = average_loss / min(100,step+1)
                print('Average loss at step %d: %f' % (step, average_loss))
                average_loss = 0
            if step % 500 == 0:
                auc = 0
                for i in range(test_batchgen.n//train_batchgen.batch_size):
                    test_batch_Xs, test_batch_Ys, test_batch_labels, test_batch_sequence_lengths = test_batchgen.next_batch()
                    test_feed_dict = {m.Xs : test_batch_Xs, m.Ys : test_batch_Ys, 
                                      m.seq_len : test_batch_sequence_lengths}
                    pred = session.run([m.predict], feed_dict=test_feed_dict)
                    auc += roc_auc_score(test_batch_labels.reshape(-1),np.array(pred).reshape(-1))/50
                print('AUC score: {}'.format(auc))   
                save_path = m.saver.save(session, 'model.ckpt')
                print('Model saved in {}'.format(save_path))



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch-size',     
                        dest='batch_size',
                        required=True,
                        type=int,
                        help="""\
                        Batch size
                        """)
    parser.add_argument('--train-steps',
                        dest='train_steps',
                        required=True,
                        type=int,
                        help='Maximum number of training steps to perform.')
    parse_args, unknown = parser.parse_known_args()
    # Set python level verbosity
    tf.logging.set_verbosity('INFO')

    if unknown:
        tf.logging.warn('Unknown arguments: {}'.format(unknown))

    train_data, id2idx =read_data_from_csv_file('0910_c_train.csv',url_i='assistments_train')
    test_data, id2idx2 =read_data_from_csv_file('0910_c_test.csv',url_i='assistments_test')

    args = parser.parse_args()
    train_batches = BatchGenerator(train_data,args.batch_size,id2idx)
    test_batches = BatchGenerator(test_data,args.batch_size,id2idx)

    s = tf.Session()

    run(s, train_batches, test_batches, args.train_steps)

