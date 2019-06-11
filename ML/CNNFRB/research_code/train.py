from __future__ import print_function
import tensorflow as tf
from resnet import RESNET, UPDATE_OPS_COLLECTION, RESNET_VARIABLES, MOVING_AVERAGE_DECAY
from image_reader import Reader, get_corpus_size, load_data_to_memory
from filterbank_reader import FReader
import math, numpy as np
import os, sys, io
import time
import pylab as plt
from sklearn.metrics import classification_report
import sklearn
from WRN_ops import lrelu

""" SCRIPT TO TRAIN PULSE DETECTION"""
MOMENTUM = 0.9

FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string('train_dir', './tmp/multi_train/',
                           """Directory where to write event logs """
                           """and checkpoint.""")
tf.app.flags.DEFINE_string('data_dir', 'Data/train/',
                           """Directory where data is located""")
tf.app.flags.DEFINE_string('tune_dir', None,
                           """Directory where finetuning data is located""")
tf.app.flags.DEFINE_string('label_file', './labels.csv',
                           """label file""")
tf.app.flags.DEFINE_string('scheme', 'molonglo', """for SP box shapes""")
tf.app.flags.DEFINE_string('dtype', 'float16', """input dtype""")
tf.app.flags.DEFINE_string('fbfilename', '/home/yunfanz/Projects/SETI/Breakthrough/FRB/Data/spliced_guppi_57991_64409_DIAG_FRB121102_0019.gpuspec.0001.8.fil',
                           """filterbank file to inference""")
tf.app.flags.DEFINE_float('learning_rate', 0.001, "learning rate.")
tf.app.flags.DEFINE_integer('batch_size', 256, "batch size")
tf.app.flags.DEFINE_integer('pool_chan', 2, "number of channels to pool together for inference")
tf.app.flags.DEFINE_integer('tpool_chan', 4, "number of times to pool together for inference")
tf.app.flags.DEFINE_integer('num_per_epoch', None, "max steps per epoch")
tf.app.flags.DEFINE_integer('epoch', 1, "number of epochs to train")
tf.app.flags.DEFINE_boolean('append', False,
                            'whether to append to output file, only for mode=predict')
tf.app.flags.DEFINE_string('relu_input', None,
                            'whether to apply relu to input')
tf.app.flags.DEFINE_boolean('resume', False,
                            'resume from latest saved state')
tf.app.flags.DEFINE_boolean('crop', False,
                            'whether to crop input')
tf.app.flags.DEFINE_boolean('is_training', True,
                            'resume from latest saved state')
tf.app.flags.DEFINE_boolean('in_memory', False,
                            'whether to train with data preloaded into memory')
tf.app.flags.DEFINE_string('mode', 'train',
                            'resume from latest saved state')
tf.app.flags.DEFINE_boolean('minimal_summaries', True,
                            'produce fewer summaries to save HD space')
#SP2_BOX = (256,342, 1)

if FLAGS.scheme == "GBTC":
    SP1_BOX = (256,10944, 1)
    SP2_BOX = (256, 342, 1)

elif FLAGS.scheme == "GBTC4chan":
    SP1_BOX = (256,2731, 1)
    SP2_BOX = (256, 342, 1)

elif FLAGS.scheme == "molonglo":
    # SP2_BOX = (256, 160, 1)
    SP2_BOX = (512, 320, 1)
    SP1_BOX = (1024,320, 1)

elif FLAGS.scheme == 'askap': 
    SP2_BOX = (512, 168, 1)
    SP1_BOX = (1024,336, 1)

def top_k_error(predictions, labels, k):
    batch_size = float(FLAGS.batch_size) #tf.shape(predictions)[0]
    in_top1 = tf.to_float(tf.nn.in_top_k(predictions, labels, k=1))
    num_correct = tf.reduce_sum(in_top1)
    return (batch_size - num_correct) / batch_size
    #return num_correct

def get_labels(img_id):
    if img_id.startswith('clean'): 
        label = 0
    elif img_id.startswith('signa'):
        label = 1
    else:
        print(img_id)
        raise Exception("labels not understood")
    return label

def test(sess, net, is_training, keep_prob, testsize=10240, outfile="./result_orig_test_null.csv"):

    if not os.path.exists(FLAGS.train_dir):
        os.makedirs(FLAGS.train_dir)
    
    coord = tf.train.Coordinator()
    reader = load_images(coord, FLAGS.data_dir, train=False)
    corpus_size = reader.corpus_size
    corpus_size = min(testsize, corpus_size)
    #import IPython; IPython.embed()
    train_batch = tf.placeholder(reader.tfdtype, name='train_batch', shape=[None, SP2_BOX[0], SP2_BOX[1], SP2_BOX[2]])
    labels = tf.placeholder(dtype=reader.label_type, shape=[None], name='label_placeholder')
    train_batch_pipe, label_pipe = reader.dequeue(FLAGS.batch_size)

    global_step = tf.get_variable('global_step', [],
                                  initializer=tf.constant_initializer(0),
                                  trainable=False)
    val_step = tf.get_variable('val_step', [],
                                  initializer=tf.constant_initializer(0),
                                  trainable=False)
    if train_batch.dtype != tf.float32:
        train_batch = tf.cast(train_batch, tf.float32)
    if FLAGS.relu_input == 'relu':
        train_batch = tf.nn.relu(train_batch) 
    elif FLAGS.relu_input == 'lrelu':
        train_batch = lrelu(train_batch, alpha=0.2)
    logits = net.inference(train_batch)
    predictions = tf.nn.softmax(logits, name='output')
    
    #top1_error = top_k_error(predictions, labels, 1)


    init = tf.global_variables_initializer()
    # import IPython; IPython.embed()
    sess.run(init)
    latest = tf.train.latest_checkpoint(FLAGS.train_dir)
    if not latest:
        print("No checkpoint to continue from in", FLAGS.train_dir)
        sys.exit(1)
    print("resume", latest)
    saver = tf.train.Saver(tf.global_variables())
    saver.restore(sess, latest)
    
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)
    reader.start_threads(sess)
    ofile = open(outfile, 'wb')
    try:
        batch_idx = corpus_size // FLAGS.batch_size
        remains = corpus_size - batch_idx*FLAGS.batch_size
        start_time = time.time()
        for idx in xrange(batch_idx):
            step = sess.run(global_step)
            #top1_error_value, y_true, y_pred = sess.run([top1_error, labels, predictions], { is_training: False, keep_prob: 1})
            #pp, ll = sess.run([predictions, labels], {is_training:False})
            #print('Predictions: ', pp)
            #print('labels: ', ll)
            tempstart = time.time()
            batch, im_ids = sess.run([train_batch_pipe, label_pipe])
            y_pred = sess.run(predictions, {train_batch:batch, is_training: False, keep_prob: 1})
            batchtime = time.time() - tempstart
            print("{} seconds per batch".format(batchtime))
            for i in xrange(FLAGS.batch_size):
                line = ','.join([im_ids[i], str(y_pred[i, 1])])+'\n'
                ofile.write(line)
        if remains > 0:
            batch, im_ids = sess.run([train_batch_pipe, label_pipe])
            y_pred = sess.run(predictions, {train_batch:batch, is_training: False, keep_prob: 1})
            for i in xrange(remains):
                line = ','.join([im_ids[i], str(y_pred[i, 1])])+'\n'
                ofile.write(line)

            # print(sklearn.metrics.classification_report(y_true,y_pred))
            # print(sklearn.metrics.confusion_matrix(y_true,y_pred))
            # print("Classification accuracy: %0.6f" % sklearn.metrics.accuracy_score(y_true,y_pred) )
            # print('weather top1 error {}'.format(top1_error_value))

    except KeyboardInterrupt:
        # Introduce a line break after ^C is displayed so save message
        # is on its own line.
        print()
        #G
    finally:
        duration = time.time() - start_time
        print('Finished, output see {}'.format(outfile))
        print('Took {} seconds to process {} files'.format(duration, corpus_size))
        coord.request_stop()
        coord.join(threads)
        ofile.close()


def train(sess, net, is_training, keep_prob, train_layers=None, fine_tune=None):

    if not os.path.exists(FLAGS.train_dir):
        os.makedirs(FLAGS.train_dir)
    
    coord = tf.train.Coordinator()
    reader = load_images(coord, FLAGS.data_dir)
    corpus_size = reader.corpus_size
    #import IPython; IPython.embed()
    if FLAGS.in_memory:
        X, Y_true = load_data_to_memory(FLAGS.data_dir,pattern='*.npy', limit=1000000, dshape=SP2_BOX)
        corpus_size = Y_true.size
    if fine_tune is not None:
        train_batch_pipe, label_pipe = reader.dequeue(FLAGS.batch_size/2)
        tX, tY_true = load_data_to_memory(FLAGS.tune_dir,pattern='*.npy', limit=100000, dshape=SP2_BOX)
        tune_size = tX.shape[0]
    else:
        train_batch_pipe, label_pipe = reader.dequeue(FLAGS.batch_size)
    train_batch = tf.placeholder(reader.tfdtype, name='train_placeholder', shape=[None, SP2_BOX[0], SP2_BOX[1], SP2_BOX[2]])
    labels = tf.placeholder(dtype=reader.label_type, shape=[None], name='label_placeholder')
    if False:
        train_batch = tf.clip_by_value(train_batch, -1, 1)
    if False: #single image normalization
        mean, var = tf.nn.moments(train_batch**2, [1], keep_dims=True)
        train_batch /= tf.sqrt(mean)
    if False:
        mean, var = tf.nn.moments(input_placeholder, [1], keep_dims=True) #single image normalization
        train_batch = tf.div(tf.subtract(input_placeholder, mean), tf.sqrt(var))
        train_batch = tf.where(tf.is_nan(train_batch), tf.zeros_like(train_batch), train_batch)
        train_batch = tf.nn.avg_pool(train_batch, 
                                ksize=[1, FLAGS.tpool_chan, FLAGS.pool_chan, 1],
                                strides=[1, FLAGS.tpool_chan, FLAGS.pool_chan, 1],
                                padding='SAME')
        if FLAGS.crop:
            train_batch = tf.image.crop_and_resize(train_batch,
                                                 boxes=[SP2_BOX])

    if train_batch.dtype != tf.float32:
        train_batch = tf.cast(train_batch, tf.float32)
    if FLAGS.relu_input == 'relu':
        train_batch = tf.nn.relu(train_batch) 
    elif FLAGS.relu_input == 'lrelu':
        train_batch = lrelu(train_batch, alpha=0.2)

    global_step = tf.get_variable('global_step', [],
                                  initializer=tf.constant_initializer(0),
                                  trainable=False)
    val_step = tf.get_variable('val_step', [],
                                  initializer=tf.constant_initializer(0),
                                  trainable=False)

    
    logits = net.inference(train_batch, name='logits')
    #import IPython; IPython.embed() 
    loss_ = net.loss(logits, labels, name='weather_loss')
    predictions = tf.nn.softmax(logits, name='output')
    #import IPython; IPython.embed()
    top1_error = top_k_error(predictions, labels, 1)

    ema = tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY, global_step)
    # loss_avg
    tf.add_to_collection(UPDATE_OPS_COLLECTION, ema.apply([loss_]))
    tf.summary.scalar('loss_avg', ema.average(loss_))

    # validation stats
    ema = tf.train.ExponentialMovingAverage(0.99, val_step)
    val_op = tf.group(val_step.assign_add(1), ema.apply([top1_error]))
    top1_error_avg = ema.average(top1_error)
    tf.summary.scalar('val_top1_error_avg', top1_error_avg)
    learning_rate = tf.placeholder(tf.float32, [], name='learning_rate')
    tf.summary.scalar('learning_rate', learning_rate)
    ###
    opt = tf.train.MomentumOptimizer(learning_rate, MOMENTUM, use_nesterov=True)
    all_grads = opt.compute_gradients(loss_)

    if not FLAGS.resume or train_layers is None:
        grads = all_grads
    else:
        grads = []
        layer_names = ['fc']
        if len(train_layers) > 0:
            layer_names += ["scale{}".format(i) for i in train_layers]
        for grad, var in all_grads:
            if any([n in var.name for n in layer_names]):
                grads.append([grad, var])

    apply_gradient_op = opt.apply_gradients(grads, global_step=global_step)
    for grad, var in grads:
        if "weight" in var.name and grad is not None and not FLAGS.minimal_summaries:
            dims = len(grad.get_shape())
            grad_per_feat = tf.reduce_mean(grad, reduction_indices=range(dims), name="avg_pool")
            tf.summary.histogram(var.op.name + '/gradients/', grad)
            tf.summary.histogram(var.op.name + '/gradients_per_feat/', grad_per_feat)

    if not FLAGS.minimal_summaries and False:
        # Display the training images in the visualizer.
        #tf.image_summary('images', images)

        for var in tf.trainable_variables():
            tf.summary.histogram(var.op.name, var)

    batchnorm_updates = tf.get_collection(UPDATE_OPS_COLLECTION)
    batchnorm_updates_op = tf.group(*batchnorm_updates)
    train_op = tf.group(apply_gradient_op, batchnorm_updates_op)

    saver = tf.train.Saver(tf.global_variables())

    summary_op = tf.summary.merge_all()

    init = tf.global_variables_initializer()
    nparams = 0
    for v in tf.global_variables():
        #sh = np.asarray(v.get_shape()).astype(np.float)
        if len(v.get_shape())>0:
            #print(v.name, int(np.prod(v.get_shape())))
            nparams += int(np.prod(v.get_shape()))
    print("Number of parameters in network", nparams)
    #import IPython; IPython.embed()
    sess.run(init)

    summary_writer = tf.summary.FileWriter(FLAGS.train_dir, sess.graph)

    if FLAGS.resume:
        latest = tf.train.latest_checkpoint(FLAGS.train_dir)
        if not latest:
            print("No checkpoint to continue from in", FLAGS.train_dir)
            sys.exit(1)
        print("resume", latest)
        saver.restore(sess, latest)

    if not FLAGS.in_memory: 
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)
        reader.start_threads(sess)
    try:
        for epoch in xrange(FLAGS.epoch):
            if FLAGS.in_memory:
                inds = np.arange(corpus_size)
                np.random.shuffle(inds)
                X, Y_true = X[inds], Y_true[inds]
            if epoch == 60:
                FLAGS.learning_rate /=  10. 
            if FLAGS.num_per_epoch:
                batch_idx = min(FLAGS.num_per_epoch, corpus_size) // FLAGS.batch_size
            else:
                batch_idx = corpus_size // FLAGS.batch_size
            for idx in xrange(batch_idx):
                start_time = time.time()

                step = sess.run(global_step)
                i = [train_op, loss_]

                write_summary = step % 100 and step > 1
                if write_summary:
                    i.append(summary_op)

                if FLAGS.in_memory:
                    inds = np.random.choice(np.arange(corpus_size), size=FLAGS.batch_size)
                    batch, batch_labels = X[inds], Y_true[inds]
                else:
                    batch, batch_labels = sess.run([train_batch_pipe, label_pipe])
                if fine_tune is not None:
                    inds = np.random.choice(np.arange(tune_size), size=FLAGS.batch_size/2)
                    tbatch, tlabels = tX[inds], tY_true[inds]
                    batch = np.vstack([batch, tbatch])
                    batch_labels = np.concatenate([batch_labels, tlabels])
                #import IPython; IPython.embed()
                o = sess.run(i, { train_batch:batch, labels:batch_labels, is_training: True, keep_prob: 0.5, learning_rate: FLAGS.learning_rate })
                #import IPython; IPython.embed()

                loss_value = o[1]
                duration = time.time() - start_time
                assert not np.isnan(loss_value), 'Model diverged with loss = NaN'

                if step % 10 == 0:
                    examples_per_sec = FLAGS.batch_size / float(duration)
                    format_str = ('Epoch %d, [%d / %d], loss = %.2f (%.1f examples/sec; %.3f '
                                  'sec/batch)')
                    print(format_str % (epoch, idx, batch_idx, loss_value, examples_per_sec, duration))

                if write_summary:
                    summary_str = o[2]
                    summary_writer.add_summary(summary_str, step)

                # Save the model checkpoint periodically.
                if step > 1 and step % 500 == 0:
                    checkpoint_path = os.path.join(FLAGS.train_dir, 'model.ckpt')
                    saver.save(sess, checkpoint_path, global_step=global_step)

                # Run validation periodically
                if step % 100 == 0:
                    _, top1_error_value, y_true, y_pred = sess.run([val_op, top1_error, labels, predictions], {train_batch:batch, labels:batch_labels, is_training: False, keep_prob: 1})
                    #pp, ll = sess.run([predictions, labels], {is_training:False})
                    #print('Predictions: ', pp)
                    #print('labels: ', ll)
                    y_pred = np.argmax(y_pred, axis=1)
                    print(sklearn.metrics.classification_report(y_true,y_pred))
                    print(sklearn.metrics.confusion_matrix(y_true,y_pred))
                    print("Classification accuracy: %0.6f" % sklearn.metrics.accuracy_score(y_true,y_pred) )
                    print('weather top1 error {}'.format(top1_error_value))

    except KeyboardInterrupt:
        # Introduce a line break after ^C is displayed so save message
        # is on its own line.
        print()
        #G
    finally:
        print('Finished, output see {}'.format(FLAGS.train_dir))
        if not FLAGS.in_memory:
            coord.request_stop()
            coord.join(threads)
        
def save_model(sess, net, is_training, keep_prob):

    input_placeholder = tf.placeholder(tf.uint8, name='input_placeholder', shape=[None, SP1_BOX[0], SP1_BOX[1], SP1_BOX[2]])
    input_32 = tf.cast(input_placeholder, tf.float32)

    mean, var = tf.nn.moments(input_32, [1], keep_dims=True) #single image normalization
    test_batch = tf.div(tf.subtract(input_32, mean), tf.sqrt(var))
    test_batch = tf.where(tf.is_nan(test_batch), tf.zeros_like(test_batch), test_batch)
    test_batch = tf.nn.avg_pool(test_batch, 
                            ksize=[1, SP1_BOX[0]/SP2_BOX[0], SP1_BOX[1]/SP2_BOX[1], 1],
                            strides=[1, SP1_BOX[0]/SP2_BOX[0], SP1_BOX[1]/SP2_BOX[1], 1],
                            padding='SAME')
    if args.scheme == 'GBTC':
        test_batch  = test_batch * 2 #trained on 4chan

    global_step = tf.get_variable('global_step', [],
                                  initializer=tf.constant_initializer(0),
                                  trainable=False)
    if FLAGS.relu_input == 'relu':
        test_batch = tf.nn.relu(test_batch) 
    elif FLAGS.relu_input == 'lrelu':
        test_batch = lrelu(test_batch, alpha=0.2)
    logits = net.inference(test_batch)
    predictions = tf.nn.softmax(logits, name='output')

    init = tf.global_variables_initializer()
    sess.run(init)
    #import IPython; IPython.embed()
    saver = tf.train.Saver(tf.global_variables())
    latest = tf.train.latest_checkpoint(FLAGS.train_dir)
    if not latest:
        print("No checkpoint to continue from in", FLAGS.train_dir)
        sys.exit(1)
    print("resume", latest)
    saver.restore(sess, latest)
    checkpoint_path = os.path.join(FLAGS.train_dir, 'model_with_preprocessing.ckpt')
    saver.save(sess, checkpoint_path, global_step=global_step)
    return

def get_predictions(weather_pred, batched=False):

    if batched:
        #print(weather_pred.shape, mpred.shape)
        string_list = []
        for n in xrange(weather_pred.shape[0]):
            string_list.append(get_predictions(weather_pred[n], batched=False))
        return string_list
    else:
        label_str= "{0:.3f},{1:.3f}".format(
            weather_pred[0], weather_pred[1])
        return label_str

def _save_pos(units, fname, tstart):
    plt.figure()
    plt.imshow(units.squeeze().T, interpolation="nearest", 
        cmap="hot", extent=[tstart, tstart+256*0.35e-3, 8, 4])
    plt.colorbar()
    plt.savefig(fname+'.png')
    plt.close()
def predict(sess, net, is_training, keep_prob, prefix='test_', append=False, from_fil=True):

    if not os.path.exists(FLAGS.train_dir):
        os.makedirs(FLAGS.train_dir)


    coord = tf.train.Coordinator()
    if from_fil:
        reader = load_filterbank(coord, FLAGS.fbfilename)
        test_batch, img_id = reader.dequeue(FLAGS.batch_size)


        
        if True:
            mean, var = tf.nn.moments(test_batch, [1], keep_dims=True) #single image normalization
            test_batch = tf.div(tf.subtract(test_batch, mean), tf.sqrt(var))
            test_batch = tf.where(tf.is_nan(test_batch), tf.zeros_like(test_batch), test_batch)
            test_batch = tf.nn.avg_pool(test_batch, 
                                    ksize=[1, FLAGS.tpool_chan, FLAGS.pool_chan, 1],
                                    strides=[1, FLAGS.tpool_chan, FLAGS.pool_chan, 1],
                                    padding='SAME')
            if FLAGS.crop:
                test_batch = tf.image.crop_and_resize(test_batch,
                                                     boxes=[SP2_BOX])


    else:
        reader = load_images(coord, FLAGS.data_dir, train=False)
        test_batch, img_id, _ = reader.dequeue(FLAGS.batch_size)
    if False:
        mean, var = tf.nn.moments(test_batch**2, [1], keep_dims=True)
        test_batch /= tf.sqrt(mean)

    corpus_size = reader.corpus_size

    global_step = tf.get_variable('global_step', [],
                                  initializer=tf.constant_initializer(0),
                                  trainable=False)

    logits = net.inference(test_batch)
    wpred = tf.nn.softmax(logits)

    
    init = tf.global_variables_initializer()
    sess.run(init)
    saver = tf.train.Saver(tf.global_variables())
    latest = tf.train.latest_checkpoint(FLAGS.train_dir)
    if not latest:
        print("No checkpoint to continue from in", FLAGS.train_dir)
        sys.exit(1)
    print("resume", latest)
    saver.restore(sess, latest)
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)
    reader.start_threads(sess)
    #import IPython; IPython.embed()
    sample_cnt = 0; add_cnt = FLAGS.batch_size
    OUTDIR = FLAGS.train_dir+'pos_5/'
    OUTFILE = FLAGS.train_dir+'pos_5.csv'
    if not append:
        outfile = open(OUTFILE, 'w')
        #outfile.write("image_name,tags\n")
    else:
        outfile = open(OUTFILE, 'a')
    detections = []
    persistent = False
    #OUTDIR = FLAGS.train_dir+'/frb20180301_3/'
    if not os.path.exists(OUTDIR):
        os.makedirs(OUTDIR)
    try:
        while True:
            print('from train', sample_cnt, corpus_size)
            if sample_cnt + FLAGS.batch_size*4 > corpus_size-1:
                break
            start_time = time.time()
            weather_scores, image_id, inputs = sess.run([wpred, img_id, test_batch], { is_training: False, keep_prob: 1 })
            string_list = get_predictions(weather_scores, batched=True)
            duration = time.time() - start_time
            #import IPython; IPython.embed()
            for n, label_str in enumerate(string_list):
                #print(prefix+str(image_id[n])+','+label_str)
                if n + sample_cnt >= corpus_size:
                    add_cnt = n 
                    break
                t_ind = int(image_id[n].split('_')[-1])
                #
                if weather_scores[n,1]> 0.5:
                    fname = prefix+str(image_id[n])#+','+label_str+'\n'
                    
                    if not persistent:
                        detections.append([t_ind, weather_scores[n][1]])
                        _save_pos(inputs[n], OUTDIR+fname, tstart=t_ind*0.0003495253)
                        #np.save(OUTDIR+fname, inputs[n])
                        print(t_ind*0.0003495253333333333, weather_scores[n][1])
                        outfile.write(','.join([str(t_ind*0.0003495253), str(weather_scores[n][1]), fname, '\n']))
                        #import IPython; IPython.embed()
                    persistent = True
                else:
                    persistent = False
                #outfile.write(prefix+str(image_id[n])+','+label_str+'\n')
            sample_cnt += add_cnt


            if sample_cnt % 20 == 0:
                perc = (FLAGS.batch_size/float(corpus_size))/(duration/(30.*60))
                qsize = sess.run(reader.queue.size())
                print("{}/{}, {} sec/batch, {} real time, queue size: {}".format(sample_cnt, corpus_size, duration, perc, qsize))
    except(ValueError):
        # Introduce a line break after ^C is displayed so save message
        # is on its own line.
        print()
    finally:
        print('Finished, output see {}'.format(fname))
        coord.request_stop()
        coord.join(threads)
        print('saving', OUTDIR+prefix)
        np.save(OUTDIR+prefix, np.asarray(detections))
        outfile.close()

def _plotNNFilter(units, fname):
    print('start')
    filters = units.shape[3]
    plt.figure(1, figsize=(20,40))
    n_columns = 6
    n_rows = min(math.ceil(filters / n_columns) + 1, 12)
    for i in range(min(filters, n_columns*n_rows)):
        plt.subplot(n_rows, n_columns, i+1)
        plt.title('Filter ' + str(i))
        plt.imshow(units[0,:,:,i], interpolation="nearest", cmap="hot")
    print('plotted', fname)
    plt.savefig(fname)
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    print('exit')
    return buf

def _get_im_summary(buf, title=None):
    image = tf.image.decode_png(buf.getvalue(), channels=3)
    image = tf.expand_dims(image, 0)
    summary_op = tf.summary.image(title, image)
    return summary_op

def save_activation(sess, net, is_training, keep_prob):

    if not os.path.exists(FLAGS.train_dir):
        os.makedirs(FLAGS.train_dir)
    
    coord = tf.train.Coordinator()
    reader = load_images(coord, FLAGS.data_dir, train=False)
    corpus_size = reader.corpus_size
    #import IPython; IPython.embed()
    test_batch, img_id = reader.dequeue(1)
    ops = []

    global_step = tf.get_variable('global_step', [],
                                  initializer=tf.constant_initializer(0),
                                  trainable=False)
    logits = net.inference(test_batch)
    init = tf.global_variables_initializer()

    sess.run(init)
    summary_writer = tf.summary.FileWriter(FLAGS.train_dir, sess.graph)
    saver = tf.train.Saver(tf.global_variables())
    latest = tf.train.latest_checkpoint(FLAGS.train_dir)
    if not latest:
        print("No checkpoint to continue from in", FLAGS.train_dir)
        sys.exit(1)
    print("resume", latest)
    saver.restore(sess, latest)
    # for op in sess.graph.get_operations():
    #     if op.name.endswith('Relu'):
    #         ops.append(op)

    #import IPython; IPython.embed()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)
    reader.start_threads(sess)

    sample_cnt = 0
    try:
        while True:
            if sample_cnt > 1:
                break
            add_cnt = 1
            iid = sess.run(img_id)
            for name, op in net.layerOutputs.items():
                #import IPython; IPython.embed()
                units = sess.run(op, { is_training: False, keep_prob: 1 })
                outname = name+'_'+iid[0]+'.png'
                plot_buf = _plotNNFilter(units,
                                 FLAGS.train_dir+'/'+outname)
                summary_op = _get_im_summary(plot_buf, title=outname)
                summary = sess.run(summary_op)
                summary_writer.add_summary(summary)

            sample_cnt += add_cnt

    finally:
        coord.request_stop()
        coord.join(threads)
        summary_writer.close()


def visualize(sess, net):

    batch = 8
    vis_shape = [batch] + list(SP2_BOX)
    #image0 = np.random.random(vis_shape)
    target0 = np.ones(batch, dtype=np.int32)
    #x = tf.placeholder(tf.float32, shape=vis_shape)
    x = tf.get_variable('input_vis',
                           shape=vis_shape,
                           initializer=tf.truncated_normal_initializer(stddev=0.5),
                           dtype=tf.float32,
                           trainable=True)
    y = tf.placeholder(tf.int32, shape=[batch])
    image_sum = tf.summary.image('vis_image', x, max_outputs=batch)
    global_step = tf.get_variable('global_step', [],
                                  initializer=tf.constant_initializer(0),
                                  trainable=False)

    logits = net.inference(x)
    opt = tf.train.AdamOptimizer(learning_rate=0.01, 
            beta1=0.9, beta2=0.999, epsilon=1e-8)
    loss = net.loss(logits, y)
    grads = opt.compute_gradients(loss, [x])
    apply_gradient_op = opt.apply_gradients(grads, global_step=global_step)

    init = tf.global_variables_initializer()

    sess.run(init)
    saver = tf.train.Saver(tf.global_variables())
    latest = tf.train.latest_checkpoint(FLAGS.train_dir)
    if not latest:
        print("No checkpoint to continue from in", FLAGS.train_dir)
        sys.exit(1)
    print("resume", latest)
    saver.restore(sess, latest)

    ite = 1000
    for i in xrange(ite):
        sess.run(apply_gradient_op, { is_training: False, keep_prob: 1 , y: target0})
    sess.run(image_sum)


def load_filterbank(coord, filename):

    reader = FReader(
        filename,
        coord,
        in_batch=128,
        queue_size=2048,
        q_shape=SP1_BOX, 
        n_threads=1
        )

    print('Using file{}'.format(filename))
    return reader


def load_images(coord, data_dir, train=True):
    if not data_dir:
        data_dir = './Data/train/'


    reader = Reader(
        data_dir,
        coord,
        pattern='*.npy',
        queue_size=8*FLAGS.batch_size, 
        min_after_dequeue=FLAGS.batch_size,
        q_shape=SP2_BOX, 
        n_threads=1,
        train=train,
        label_file=None,
        pref=True,
        dtype=FLAGS.dtype
        )

    print('Using data_dir{}, size {}'.format(data_dir, reader.corpus_size))
    return reader


def main(_):

    sessconfig = tf.ConfigProto()
    sessconfig.gpu_options.allow_growth = False
    sess = tf.Session(config=sessconfig)
    #sess = tf.Session(config=tf.ConfigProto(log_device_placement=False))
    
    is_training = tf.placeholder('bool', [], name='is_training')
    keep_prob = tf.placeholder(tf.float32, [], name='keep_prob')
    # for resnet 101: num_blocks=[3, 4, 23, 3]
    # for resnet 152: num_blocks=[3, 8, 36, 3]

    net = RESNET(sess,
               dim=2,
               num_classes=2,
               num_blocks=[1, 2, 3, 2],  # first chan is not a block
               num_chans=[32,32,64,128,256],
               use_bias=False, # defaults to using batch norm
               bottleneck=False,
               is_training=is_training)

    print(FLAGS.mode)
    if FLAGS.mode == "train":
        train(sess, net, is_training, keep_prob, fine_tune=FLAGS.tune_dir)
    if FLAGS.mode == "test":
        test(sess, net, is_training, keep_prob)
    elif FLAGS.mode == 'predict':
        pref = os.path.basename(FLAGS.fbfilename).split('.')[0]
        predict(sess, net, is_training, keep_prob, prefix=pref, append=FLAGS.append)
    elif FLAGS.mode == "visualize":
        #visualize(sess, net)
        save_activation(sess, net, is_training, keep_prob)
    elif FLAGS.mode == "savemodel":
        #visualize(sess, net)
        save_model(sess, net, is_training, keep_prob)


if __name__ == '__main__':
    tf.app.run()
