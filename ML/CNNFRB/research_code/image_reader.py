from __future__ import print_function
import fnmatch
import os
import re
import threading
from scipy import ndimage
from skimage import measure
import pandas
import numpy as np
import tensorflow as tf
import skimage
import skimage.io
import skimage.transform
import os
import zipfile


LABEL_FILE = "./labels.csv"

#G
def get_corpus_size(directory, pattern='*.png'):
    '''Recursively finds all files matching the pattern.'''
    files = []
    for root, dirnames, filenames in os.walk(directory):
        for filename in fnmatch.filter(filenames, pattern):
            files.append(os.path.join(root, filename))
    return len(files)

#def get_imgid(fpath):
#    return int(fpath.split('/')[-1].split('.')[0])
def get_imgid(filename, pref):
    if pref:
        tsplits = filename.split('/')
        img_id = '_'.join([tsplits[-2], tsplits[-1]]).split('.')[0]
    else:
        img_id = '.'.join(filename.split('/')[-1].split('.')[:-1])
    return img_id
def find_files(directory, pattern='*.png', sortby="shuffle"):
    '''Recursively finds all files matching the pattern.'''
    files = []
    for root, dirnames, filenames in os.walk(directory):
        for filename in fnmatch.filter(filenames, pattern):
            files.append(os.path.join(root, filename))

    if sortby == 'auto':
        files = np.sort(files)
    elif sortby == 'shuffle':
        np.random.shuffle(files)
    return files


def _augment_img(img):
    rands = np.random.random(3)
    if rands[0] < 0.5:
        img = np.fliplr(img)
    if rands[1] < 0.5:
        img = np.flipud(img)
    return img

def load_image(directory, pattern='*.png', train=False, pref=False, select=0.99, dtype=np.float32):
    '''Generator that yields pixel_array from dataset, and
    additionally the ID of the corresponding patient.'''
    if train:
        sort_by = 'shuffle'
    else:
        sort_by = 'auto'
    files = find_files(directory, pattern=pattern, sortby=sort_by)
    csize = int(select*len(files))
    files = np.random.choice(files, size=csize)
    for filename in files:
        if pattern == '*.png':
            img = skimage.io.imread(filename).astype(dtype)
            img /= 256.
        elif pattern == '*.npy':
            img = np.load(filename).astype(dtype)
        elif pattern == '*.npz':
            img = np.load(filename)['frame'].astype(dtype)
            if False:
                img *= np.load(filename)['mask'].astype(dtype)
        if train:
            img = img.T[...,np.newaxis]
        img_id = get_imgid(filename, pref=pref)
        #print(filename, img.shape)
        yield img, img_id

def load_data_to_memory(directory, pattern='*.npy', train=True, pref=False, limit=100000, dtype=np.float16, dshape=None):
    if train:
        sort_by = 'shuffle'
    else:
        sort_by = 'auto'
    
    files = find_files(directory, pattern=pattern, sortby=sort_by)
    print("Loading {} files into memory".format(min(len(files), limit)))
    if limit < len(files):
        files = files[:limit]
    Y_true = []
    X = np.zeros((len(files),)+dshape, dtype=dtype)
    for i, filename in enumerate(files):
        if i % 1000 == 0:
            print(i)
        if pattern == '*.png':
            img = skimage.io.imread(filename)
            img /= 256.
        elif pattern == '*.npy':
            img = np.load(filename)
        elif pattern == '*.npz':
            img = np.load(filename)['frame']
            if False:
                img *= np.load(filename)['mask']
        if train:
            img = img.T[...,np.newaxis]
        if img.dtype is not X.dtype:
            img = img.astype(X.dtype)
        X[i] = img
        #img_id = get_imgid(filename, pref=pref)
        if "signa" not in filename:
            Y_true.append(0)
        elif "signa" in filename:
            Y_true.append(1)
        else:
            print(filename + " not understood")
    #X = np.stack(X, axis=0)
    return X, np.asarray(Y_true)
        
def convert_to_chunck(directory, outdir, batch_size=512, pattern='*.npy', train=True, pref=False, limit=1000000, dtype=np.float16, dshape=None):
    if train:
        sort_by = 'shuffle'
    else:
        sort_by = 'auto'
    
    files = find_files(directory, pattern=pattern, sortby=sort_by)
    print("Loading {} files into memory".format(min(len(files), limit)))
    if limit < len(files):
        files = files[:limit]
    Y_true = []
    X = np.zeros((batch_size,)+dshape, dtype=dtype)
    for i, filename in enumerate(files):
        if len(files) - i < batch_size:
            X = X[:len(files) - i]
        if i % batch_size == 0:
            print(i)
            np.savez(outdir)
        if pattern == '*.png':
            img = skimage.io.imread(filename)
            img /= 256.
        elif pattern == '*.npy':
            img = np.load(filename)
        elif pattern == '*.npz':
            img = np.load(filename)['frame']
            if False:
                img *= np.load(filename)['mask']
        if train:
            img = img.T[...,np.newaxis]
        if img.dtype is not X.dtype:
            img = img.astype(X.dtype)
        X[i] = img
        #img_id = get_imgid(filename, pref=pref)
        if "clean" in filename:
            Y_true.append(0)
        elif "signa" in filename:
            Y_true.append(1)
        else:
            print(filename + " not understood")
    #X = np.stack(X, axis=0)
    return X, np.asarray(Y_true)

def load_label_df(filename):
    df_train = pandas.DataFrame.from_csv(filename, index_col='fname')
    #import IPython; IPython.embed()
    #df_train['label'] = df_train['DM']#.apply(lambda row: get_weather(row))
    print(df_train.columns)
    return df_train
# def get_loss_weights(dframe=None, label_file=LABEL_FILE):
#     if dframe is None:
#         dframe = load_label_df(label_file)
#     keys, counts = np.unique(dframe['label'], return_counts=True)
#     weights = counts.astype(np.float32)/np.sum(counts)
#     return dict(zip(keys, weights))

class Reader(object):
    '''Generic background reader that preprocesses files
    and enqueues them into a TensorFlow queue.'''

    def __init__(self,
                 data_dir,
                 coord,
                 train=True, 
                 threshold=None,
                 queue_size=16, 
                 min_after_dequeue=4,
                 q_shape=None,
                 pattern='*.npy', 
                 n_threads=1,
                 multi=True,
                 label_file=LABEL_FILE,
                 label_type=tf.int32,
                 pref=False,
                 dtype=np.float32):
        self.data_dir = data_dir
        self.coord = coord
        self.n_threads = n_threads
        self.threshold = threshold
        self.ftype = pattern
        self.corpus_size = get_corpus_size(self.data_dir, pattern=self.ftype)
        self.threads = []
        self.q_shape = q_shape
        self.multi = multi
        self.train = train
        self.npdtype = dtype
        self.tfdtype = tf.as_dtype(self.npdtype)
        self.sample_placeholder = tf.placeholder(dtype=self.tfdtype, shape=None)
        self.label_shape = []
        self.label_type = label_type
        self.pattern = pattern
        self.pref = pref
        self.labels_df = None
        self.label_shape = []
        if self.train:
            if label_file is not None:
                self.labels_df = load_label_df(label_file)
                self.label_shape = [len(self.labels_df.columns)]
            
            self.label_placeholder = tf.placeholder(dtype=self.label_type, shape=self.label_shape, name='label') #!!!
            if self.q_shape:
                #self.queue = tf.FIFOQueue(queue_size,[tf.float32,tf.int32], shapes=[q_shape,[]])

                self.queue = tf.RandomShuffleQueue(queue_size, min_after_dequeue,
                    [self.tfdtype,label_type], shapes=[q_shape, self.label_shape])
            else:
                self.q_shape = [(1, None, None, 1)]
                self.queue = tf.PaddingFIFOQueue(queue_size,
                                                 [self.tfdtype, label_type],
                                                 shapes=[self.q_shape,self.label_shape])
        else:
            self.label_placeholder = tf.placeholder(dtype=tf.string, shape=[], name='label') #!!!
            self.queue = tf.FIFOQueue(queue_size,[self.tfdtype,tf.string], shapes=[q_shape,[]])
        self.enqueue = self.queue.enqueue([self.sample_placeholder, self.label_placeholder])

    def dequeue(self, num_elements):
        images, labels = self.queue.dequeue_many(num_elements)
        #print(labels[:4])
        return images, labels

    def thread_main(self, sess):
        buffer_ = np.array([])
        stop = False
        # Go through the dataset multiple times
        while not stop:
            iterator = load_image(self.data_dir, train=self.train, pattern=self.pattern, pref=self.pref, dtype=self.npdtype)
            for img, img_id in iterator:
                #print(filename)
                if self.train:	
                    if self.labels_df is not None:
                        try: 
                            label = [self.labels_df[c][img_id] for c in self.labels_df.columns]
                        
                        except(KeyError):
                            print('No match for ', img_id)
                            continue
                    else:
                        if img_id.startswith('clean'): 
                            label = 0
                        elif img_id.startswith('signa'):
                            label = 1
                        else:
                            print(img_id)
                            raise Exception("labels not understood")
                else:
                    label = img_id
                if self.coord.should_stop():
                    stop = True
                    break
                if self.threshold is not None:
                    #TODO:  Perform quality check if needed
                    pass
                #print(img.shape, self.q_shape)
                if img.shape != self.q_shape:
                    img = img.T[...,np.newaxis]
                sess.run(self.enqueue,
                             feed_dict={self.sample_placeholder: img, self.label_placeholder: label})
                    
    def start_threads(self, sess):
        for _ in xrange(self.n_threads):
            thread = threading.Thread(target=self.thread_main, args=(sess,))
            thread.daemon = True  # Thread will close when parent quits.
            thread.start()
            self.threads.append(thread)
        return self.threads
