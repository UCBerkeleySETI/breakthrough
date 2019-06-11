import tensorflow as tf, numpy as np
import argparse 
from sigpyproc.Readers import FilReader
import os, fnmatch
from time import time
from skimage import measure

"""Inference script for Breakthrough Listen observations of Fast Radio Bursts
   Type:       Single Beam
   Instrument: Green Bank Telescope

   Author: Yunfan Gerry Zhang
           Breakthrough Listen
"""

parser = argparse.ArgumentParser()
parser.add_argument("--model", default="./models/GBT_Cband.pb", type=str, help="Frozen model file to import")
parser.add_argument("--filterbank_dir", default="/data2/molonglo/", type=str, help="Directory containing filterbanks")
parser.add_argument("--threshold", default=0.9, type=float, help="confident threshold of detection")
parser.add_argument("--batchsize", default=256, type=int, help="batch size for inference")
args = parser.parse_args()

def find_files(directory, pattern='*.png', sortby="shuffle", flag=None):
    '''Recursively finds all files matching the pattern.'''
    files = []
    for root, dirnames, filenames in os.walk(directory):
        for filename in fnmatch.filter(filenames, pattern):
            path = os.path.join(root, filename)

            files.append(path)

    if sortby == 'auto':
        files = np.sort(files)
    elif sortby == 'shuffle':
        np.random.shuffle(files)
    return files

def load_graph(frozen_graph_filename):
    """ Function to load frozen TensorFlow graph"""
    with tf.gfile.GFile(frozen_graph_filename, "rb") as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
    with tf.Graph().as_default() as graph:
        tf.import_graph_def(graph_def, name="prefix")
    return graph

def get_readers(fil_files, nbeams=16):
    """Load blimpy Waterfall objects for filterbank file reading"""
    fils = []
    for f in sorted(fil_files)[:nbeams]:
        fils.append(FilReader(f))
    return fils

def read_input(reader, t0, NT, a=None, tstep=256, batchsize=128, nchan=10944):
    """Read a chunck of data from each beam
    output:
    array of shape (batchsize, tstep, nchan, 1)
    """
    u8 = (reader.header['nbits'] == 8)
    if a is None:
        a = np.zeros((batchsize, tstep, nchan, 1), dtype=np.uint8)

    a = reader.readBlock(start=t0, nsamps=min(tstep*batchsize, NT-t0)).T
    a = a[:,1190:1190+nchan]  #get 4 to 8 GHz
    to_mask = 0
    if a.shape[0] < tstep*batchsize:  # pad with zeros
        to_mask = int((tstep*batchsize - a.shape[0])/tstep)
        a = np.concatenate([a, np.zeros((tstep*batchsize-a.shape[0], nchan), dtype=a.dtype)], axis=0)
    assert a.shape == (tstep*batchsize, nchan)
    a = a.reshape((batchsize, tstep, nchan, 1))

    return a, to_mask

if __name__ == '__main__':

    graph = load_graph(args.model)
    TSTEP = 256*args.batchsize #window of time stamps

    # We access the input and output nodes 
    is_training = graph.get_tensor_by_name('prefix/is_training:0')
    x = graph.get_tensor_by_name('prefix/input_placeholder:0')
    y = graph.get_tensor_by_name('prefix/output:0')


    files = find_files(args.filterbank_dir, pattern='*.fil')
    files = sorted(files)
    reader = get_readers(files, 1)[0]
    NT = reader.header['nsamples']
    dt = reader.header['tsamp']

    print('sampling time', dt)
    
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    with tf.Session(graph=graph, config=config) as sess:
        t0 = 0
        a = None
        step = 0
        while t0 < NT:
            a, to_mask = read_input(reader, t0, NT, a=a)
            t0 += TSTEP; step += 1
            start = time()
            y_out = sess.run(y, feed_dict={ x: a, is_training:False })
            duration = time() - start
            if step % 10 == 0:
                speed = dt*TSTEP/duration
                print'{} / {},  speed: {} times real time'.format(t0,NT, speed)
            scores = y_out[:,1].copy()
            detections = scores > args.threshold
            if to_mask != 0:
                detections[-to_mask:] = False
            ndetections = np.sum(detections)
            if ndetections > 0:
                frames_with_detection = np.asarray([ind for ind, val in enumerate(detections) if val])
                print("Detection at TOA",(t0/256+frames_with_detection)*dt, scores[frames_with_detection])
