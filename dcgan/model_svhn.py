import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from utils import leaky_relu, view_samples, load_checkpoint


class Config(object):
    """
    Stores model hyperparameters and data information
    Model objects are passed a Config() object at instantiation
    """
    batch_size = 128
    real_dim = (32,32,3)
    z_dim = 100
    sample_size = 16
    num_epoch = 25
    alpha = 0.2
    lr = 2e-4
    beta1 = 0.5
    checkpoint_dir = 'checkpoint/svhn/'
    sample_dir = 'train/svhn/'
    
class GAN(object):
    def __init__(self, config):
        """
        Initializes the model
        :param config: A model configuration object of type Config
        """
        #tf.reset_default_graph()
        self.config = config
        self.input_real, self.input_z = model_inputs(self.config.real_dim, self.config.z_dim)
        
        G_model = generator(self.input_z, alpha=self.config.alpha)
        logits_real = discriminator(self.input_real, alpha=self.config.alpha)
        logits_fake = discriminator(G_model, alpha=self.config.alpha, reuse=True)
        self.D_loss, self.G_loss = model_loss(logits_real, logits_fake)
        self.D_opt, self.G_opt = model_opt(self.D_loss, self.G_loss, self.config.lr, self.config.beta1)

    def train(self, sess, svhn, show_every=250, print_every=50):
        """Train a GAN for a certain number of epochs"""
        saver = tf.train.Saver()
        sample_z = np.random.uniform(-1, 1, size=(self.config.sample_size, self.config.z_dim))

        losses = []
        steps = 0

        start_epoch = load_checkpoint(sess, saver, self.config.checkpoint_dir)
        for e in range(start_epoch, self.config.num_epoch):
            for minibatch, minibatch_y in svhn.batches(self.config.batch_size):
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
                    gen_samples = sess.run(generator(self.input_z, alpha=self.config.alpha, reuse=True, training=False),
                                           feed_dict={self.input_z: sample_z})
                    _ = view_samples(gen_samples, 4, 4)
                    plt.savefig(os.path.join(self.config.sample_dir, 'train_{}.png'.format(steps)))
                    plt.show()
            
            # save model after each epoch
            saver.save(sess, os.path.join(self.config.checkpoint_dir, 'GAN_epoch{}.ckpt'.format(e)))

        print('Final images')
        gen_samples = sess.run(generator(self.input_z, alpha=self.config.alpha, reuse=True, training=False), feed_dict={self.input_z: sample_z})
        _ = view_samples(gen_samples, 4, 4)
        plt.savefig(os.path.join(self.config.sample_dir, 'final.png'))
        plt.show()
        
        # save final model
        saver.save(sess, os.path.join(self.config.checkpoint_dir, 'GAN_final.ckpt'))
        return losses

def model_inputs(real_dim, z_dim):
    inputs_real = tf.placeholder(tf.float32, (None, *real_dim), name='input_real')
    inputs_z = tf.placeholder(tf.float32, (None, z_dim), name='input_z')
    return inputs_real, inputs_z        

def discriminator(x, alpha=0.2, reuse=False):
    with tf.variable_scope('discriminator', reuse=reuse):
        # Input layer is 32x32x3
        x1 = tf.layers.conv2d(x, 64, 5, strides=2, padding='same')
        x1 = leaky_relu(x1, alpha=alpha)
        # 16x16x64
        
        x2 = tf.layers.conv2d(x1, 128, 5, strides=2, padding='same')
        x2 = tf.layers.batch_normalization(x2, training=True)
        x2 = leaky_relu(x2, alpha=alpha)
        # 8x8x128
        
        x3 = tf.layers.conv2d(x2, 256, 5, strides=2, padding='same')
        x3 = tf.layers.batch_normalization(x3, training=True)
        x3 = leaky_relu(x3, alpha=alpha)
        # 4x4x256

        # Flatten it
        x3_flat = tf.reshape(x3, (-1, 4*4*256))
        logits = tf.layers.dense(x3_flat, 1)
        return logits

def generator(z, output_dim=3, alpha=0.2, reuse=False, training=True):
    with tf.variable_scope('generator', reuse=reuse):
        x1 = tf.layers.dense(z, 4*4*512)
        x1 = tf.reshape(x1, (-1, 4, 4, 512))
        x1 = tf.layers.batch_normalization(x1, training=training)
        x1 = leaky_relu(x1, alpha=alpha)
        # 4x4x512 now
        
        x2 = tf.layers.conv2d_transpose(x1, 256, 5, strides=2, padding='same')
        x2 = tf.layers.batch_normalization(x2, training=training)
        x2 = leaky_relu(x2, alpha=alpha)
        # 8x8x256 now
        
        x3 = tf.layers.conv2d_transpose(x2, 128, 5, strides=2, padding='same')
        x3 = tf.layers.batch_normalization(x3, training=training)
        x3 = leaky_relu(x3, alpha=alpha)
        # 16x16x128 now
        
        # Output layer
        logits = tf.layers.conv2d_transpose(x3, output_dim, 5, strides=2, padding='same')
        # 32x32x3 now
        
        img = tf.tanh(logits)
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
