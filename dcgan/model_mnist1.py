import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from utils import preprocess_img, leaky_relu, show_images


class Config(object):
    """
    Stores model hyperparameters and data information
    Model objects are passed a Config() object at instantiation
    """
    batch_size = 128
    real_dim = (784,)
    z_dim = 96
    sample_size = 16
    num_epoch = 1
    lr = 1e-3
    beta1 = 0.5
    checkpoint_dir = 'checkpoint/mnist1/'
    sample_dir = 'train/mnist1/'
    
class GAN(object):
    def __init__(self, config):
        """
        Initializes the model
        :param config: A model configuration object of type Config
        """
        #tf.reset_default_graph()
        self.config = config
        self.input_real, self.input_z = model_inputs(self.config.real_dim, self.config.z_dim)
        
        G_model = generator(self.input_z)
        logits_real = discriminator(preprocess_img(self.input_real))
        logits_fake = discriminator(G_model, reuse=True)
        self.D_loss, self.G_loss = model_loss(logits_real, logits_fake)
        self.D_opt, self.G_opt = model_opt(self.D_loss, self.G_loss, self.config.lr, self.config.beta1)

    def train(self, sess, mnist, show_every=250, print_every=50):
        """Train a GAN for a certain number of epochs"""
        saver = tf.train.Saver()
        sample_z = np.random.uniform(-1, 1, size=(self.config.sample_size, self.config.z_dim))

        losses = []
        steps = 0

        start_epoch = 0 #load_checkpoint(sess, saver, self.config.checkpoint_dir)
        for e in range(start_epoch, self.config.num_epoch):
            max_iter = int(mnist.train.num_examples/self.config.batch_size)
            for _ in range(max_iter):
                minibatch, minibatch_y = mnist.train.next_batch(self.config.batch_size)
                steps += 1

                # Sample random noise for G
                batch_z = np.random.uniform(-1, 1, size=(self.config.batch_size, self.config.z_dim))
                
                # Run optimizers
                _, D_loss_curr = sess.run([self.D_opt, self.D_loss], feed_dict={self.input_real: minibatch, self.input_z: batch_z})
                _, G_loss_curr = sess.run([self.G_opt, self.G_loss], feed_dict={self.input_z: batch_z, self.input_real: minibatch})
                losses.append([D_loss_curr, G_loss_curr])

                # print loss every so often
                # We want to make sure D_loss doesn't go to 0
                if steps % print_every == 0:
                    print('... Epoch {}, Iter {}, D:{:.4}, G:{:.4}'.format(e, steps, D_loss_curr, G_loss_curr))

                # show sample result every so often 
                if steps % show_every == 1:
                    gen_samples = sess.run(generator(self.input_z, reuse=True, training=False),
                                           feed_dict={self.input_z: sample_z})
                    show_images(gen_samples)
                    plt.savefig(os.path.join(self.config.sample_dir, 'train_{}.png'.format(steps)))
                    plt.show()
            
            # save model after each epoch
            saver.save(sess, os.path.join(self.config.checkpoint_dir, 'GAN_epoch{}.ckpt'.format(e)))

        print('Final images')
        gen_samples = sess.run(generator(self.input_z, reuse=True, training=False), feed_dict={self.input_z: sample_z})
        show_images(gen_samples)
        plt.savefig(os.path.join(self.config.sample_dir, 'final.png'))
        plt.show()
        
        # save final model
        saver.save(sess, os.path.join(self.config.checkpoint_dir, 'GAN_final.ckpt'))
        return losses

def model_inputs(real_dim, z_dim):
    inputs_real = tf.placeholder(tf.float32, (None, *real_dim), name='input_real')
    inputs_z = tf.placeholder(tf.float32, (None, z_dim), name='input_z')
    return inputs_real, inputs_z        

def discriminator(x, reuse=False):
    with tf.variable_scope("discriminator", reuse=reuse):
        x_img = tf.reshape(x, [-1,28,28,1])
        conv1 = tf.layers.conv2d(x_img, 32, 5, activation=leaky_relu, padding='valid')
        pool1 = tf.layers.max_pooling2d(conv1, 2, 2, padding='same')
        conv2 = tf.layers.conv2d(pool1, 64, 5, activation=leaky_relu, padding='valid')
        pool2 = tf.layers.max_pooling2d(conv2, 2, 2, padding='same')
        pool2_flat = tf.reshape(pool2, [-1,1024])
        fc1 = tf.layers.dense(pool2_flat, 1024, activation=leaky_relu)
        logits = tf.layers.dense(fc1, 1)
        return logits

def generator(z, reuse=False, training=True):
    with tf.variable_scope("generator", reuse=reuse):
        fc1 = tf.layers.dense(z, 1024, activation=tf.nn.relu)
        bn1 = tf.layers.batch_normalization(fc1, training=training)
        fc2 = tf.layers.dense(bn1, 7*7*128, activation=tf.nn.relu)
        bn2 = tf.layers.batch_normalization(fc2, training=training)
        bn2_img = tf.reshape(bn2, [-1,7,7,128])
        convT1 = tf.layers.conv2d_transpose(bn2_img, 64, 4, strides=2, activation=tf.nn.relu, padding='same')
        bn3 = tf.layers.batch_normalization(convT1, training=training)
        convT2 = tf.layers.conv2d_transpose(bn3, 1, 4, strides=2, padding='same')
        img = tf.tanh(convT2)
        img = tf.reshape(img, [-1,784])
        return img

def model_loss(logits_real, logits_fake):
    """
    Compute the GAN loss
    :param logits_real: Tensor, shape [batch_size, 1], output of discriminator
        Log probability that the image is real for each real image
    :param logits_fake: Tensor, shape[batch_size, 1], output of discriminator
        Log probability that the image is real for each fake image
    :return: A tuple of (discriminator loss, generator loss)
    """
    D_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
        logits=logits_real, labels=tf.ones_like(logits_real)))
    D_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits( 
        logits=logits_fake,labels=tf.zeros_like(logits_fake)))
    D_loss = D_loss_real + D_loss_fake
    G_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
        logits=logits_fake, labels=tf.ones_like(logits_fake)))
    return D_loss, G_loss

def model_opt(D_loss, G_loss, learning_rate, beta1):
    """
    Get optimization operations
    :param D_loss: Discriminator loss Tensor
    :param G_loss: Generator loss Tensor
    :param learning_rate: Learning Rate Placeholder
    :param beta1: The exponential decay rate for the 1st moment in the optimizer
    :return: A tuple of (discriminator training operation, generator training operation)
    """
    # Get weights and bias to update
    #t_vars = tf.trainable_variables()
    #d_vars = [var for var in t_vars if var.name.startswith('discriminator')]
    #g_vars = [var for var in t_vars if var.name.startswith('generator')]
    # Get the list of variables for the discriminator and generator
    D_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'discriminator')
    G_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'generator')
        
    # Optimize
    with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
        D_train_opt = tf.train.AdamOptimizer(learning_rate, beta1=beta1).minimize(D_loss, var_list=D_vars)
        G_train_opt = tf.train.AdamOptimizer(learning_rate, beta1=beta1).minimize(G_loss, var_list=G_vars)
    return D_train_opt, G_train_opt
