import os
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

###################################################################################
# Common utility functions
#-------------------------
def show_images(images):
    plt.rcParams['figure.figsize'] = (10.0, 8.0) # set default size of plots
    plt.rcParams['image.interpolation'] = 'nearest'
    plt.rcParams['image.cmap'] = 'gray'

    images = np.reshape(images, [images.shape[0], -1])  # images reshape to (batch_size, D)
    sqrtn = int(np.ceil(np.sqrt(images.shape[0])))
    sqrtimg = int(np.ceil(np.sqrt(images.shape[1])))

    fig = plt.figure(figsize=(sqrtn, sqrtn))
    gs = gridspec.GridSpec(sqrtn, sqrtn)
    gs.update(wspace=0.05, hspace=0.05)

    for i, img in enumerate(images):
        ax = plt.subplot(gs[i])
        plt.axis('off')
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_aspect('equal')
        plt.imshow(img.reshape([sqrtimg,sqrtimg]))

def view_samples(images, nrows, ncols, figsize=(5,5)):
    fig, axes = plt.subplots(figsize=figsize, nrows=nrows, ncols=ncols, 
                             sharey=True, sharex=True)
    for ax, img in zip(axes.flatten(), images):
        ax.axis('off')
        img = ((img - img.min())*255 / (img.max() - img.min())).astype(np.uint8)
        ax.set_adjustable('box-forced')
        im = ax.imshow(img, aspect='equal')
   
    plt.subplots_adjust(wspace=0, hspace=0)
    return fig, axes

def load_checkpoint(sess, saver, checkpoint_dir):
    import re
    print('=== Reading checkpoints ===')
    ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
    if ckpt and ckpt.model_checkpoint_path:
        ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
        saver.restore(sess, os.path.join(checkpoint_dir, ckpt_name))
        print('Successful in reading {}'.format(ckpt_name))
        e = int(next(re.finditer('(\d+)(?!.*\d)', ckpt_name)).group(0))
        return e + 1
    else:
        print('Failed to find a checkpoint')
        return 0

def count_params():
    """Count the number of parameters in the current TensorFlow graph """
    param_count = np.sum([np.prod(x.get_shape().as_list()) for x in tf.global_variables()])
    return param_count

def get_session():
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    session = tf.Session(config=config)
    return session

def leaky_relu(x, alpha=0.01):
    """Compute the leaky ReLU activation function.
    
    Args:
    :param x: TensorFlow Tensor with arbitrary shape
    :param alpha: leak parameter for leaky ReLU
    
    Returns:
    TensorFlow Tensor with the same shape as x
    """
    return tf.maximum(alpha*x, x)

###################################################################################
# Pertaining to mnist
#-------------------------
def preprocess_img(x):
    return 2 * x - 1.0

def deprocess_img(x):
    return (x + 1.0) / 2.0

def load_mnist():
    """Wrapper function to load mnist data"""
    from tensorflow.examples.tutorials.mnist import input_data
    mnist = input_data.read_data_sets('data/mnist/', one_hot=False)
    return mnist

def convert_y_vec(y, y_dim=10):
    y_vec = np.zeros((len(y), y_dim), dtype=np.float32)
    for i, label in enumerate(y):
        y_vec[i, y[i]] = 1.
    return y_vec

###################################################################################
# Pertaining to svhn
#-------------------------
def scale(x, feature_range=(-1, 1)):
    # scale to (0, 1)
    x = ((x - x.min())/(255 - x.min()))
    
    # scale to feature_range
    mmin, mmax = feature_range
    x = x * (mmax - mmin) + mmin
    return x

class Dataset_svhn:
    def __init__(self, train, test, val_frac=0.5, shuffle=False, scale_func=None):
        split_idx = int(len(test['y'])*(1 - val_frac))
        self.test_x, self.valid_x = test['X'][:,:,:,:split_idx], test['X'][:,:,:,split_idx:]
        self.test_y, self.valid_y = test['y'][:split_idx], test['y'][split_idx:]
        self.train_x, self.train_y = train['X'], train['y']
        
        self.train_x = np.rollaxis(self.train_x, 3)
        self.valid_x = np.rollaxis(self.valid_x, 3)
        self.test_x = np.rollaxis(self.test_x, 3)
        
        if scale_func is None:
            self.scaler = scale
        else:
            self.scaler = scale_func
        self.shuffle = shuffle
        
    def batches(self, batch_size):
        if self.shuffle:
            idx = np.arange(len(self.train_x))
            np.random.shuffle(idx)
            self.train_x = self.train_x[idx]
            self.train_y = self.train_y[idx]
        
        n_batches = len(self.train_y)//batch_size
        for ii in range(0, len(self.train_y), batch_size):
            x = self.train_x[ii:ii+batch_size]
            y = self.train_y[ii:ii+batch_size]
            
            yield self.scaler(x), y

def load_svhn():  
    from urllib.request import urlretrieve
    from os.path import isfile
    from tqdm import tqdm
    from scipy.io import loadmat

    data_dir = 'data/svhn/'

    class DLProgress(tqdm):
        last_block = 0

        def hook(self, block_num=1, block_size=1, total_size=None):
            self.total = total_size
            self.update((block_num - self.last_block) * block_size)
            self.last_block = block_num

    if not isfile(data_dir + "train_32x32.mat"):
        with DLProgress(unit='B', unit_scale=True, miniters=1, desc='SVHN Training Set') as pbar:
            urlretrieve(
                'http://ufldl.stanford.edu/housenumbers/train_32x32.mat',
                data_dir + 'train_32x32.mat',
                pbar.hook)

    if not isfile(data_dir + "test_32x32.mat"):
        with DLProgress(unit='B', unit_scale=True, miniters=1, desc='SVHN Testing Set') as pbar:
            urlretrieve(
                'http://ufldl.stanford.edu/housenumbers/test_32x32.mat',
                data_dir + 'test_32x32.mat',
                pbar.hook)
        
    trainset = loadmat(data_dir + 'train_32x32.mat')
    testset = loadmat(data_dir + 'test_32x32.mat')
    return trainset, testset

###################################################################################
# Pertaining to celeba
#-------------------------
import skimage.io
import skimage.transform

def rescale(img, reverse=False):
    if reverse:
        return (img + 1) / 2
    return img * 2 - 1

def centre_crop(img, output_shape):
    h, w = img.shape[:2]
    crop_h, crop_w = output_shape
    j = int(round((h-crop_h)/2.))
    i = int(round((w-crop_w)/2.))
    return img[j:j+crop_h, i:i+crop_w]

def get_image(path, as_grey=False, crop_shape=None, resize_shape=(64,64)):
    img = skimage.io.imread(path, as_grey=as_grey)
    if crop_shape is not None:
        img = centre_crop(img, crop_shape)
    img_resize = skimage.transform.resize(img, resize_shape, mode='reflect')
    img_rescale = rescale(img_resize)
    return img_rescale

class Dataset_celeba:
    def __init__(self, shuffle=False):
        from glob import glob
        self.data = glob('data/celeba/*.jpg')
        self.shuffle = shuffle
        
    def batches(self, batch_size):
        if self.shuffle:
            idx = np.arange(len(self.data))
            np.random.shuffle(idx)
            self.data = self.data[idx]
        
        n_batches = len(self.data)//batch_size
        for ii in range(0, len(self.data), batch_size):
            batch_files = self.data[ii:ii+batch_size]
            batch = [get_image(batch_file, crop_shape=(128,128)) for batch_file in batch_files]

            yield batch
