import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from utils import convert_y_vec, preprocess_img, leaky_relu, show_images


class Config(object):
    """
    Stores model hyperparameters and data information
    Model objects are passed a Config() object at instantiation
    """
    batch_size = 128
    real_dim = (784,)
    z_dim = 100
    y_dim = 10
    sample_size = 16
    num_epoch = 2
    lr = 1e-3
    beta1 = 0.5
    checkpoint_dir = 'checkpoint/mnist3/'
    sample_dir = 'train/mnist3/'
    
class GAN(object):
    def __init__(self, config):
        """
        Initializes the model
        :param config: A model configuration object of type Config
        """
        self.config = config
        self.input_real, self.input_z, self.input_y = model_inputs(self.config.real_dim, self.config.z_dim, self.config.y_dim)

        G_model = generator(self.input_z, self.input_y, self.config.batch_size, self.config.y_dim)
        logits_real = discriminator(preprocess_img(self.input_real), self.input_y, self.config.batch_size, self.config.y_dim)
        logits_fake = discriminator(G_model, self.input_y, self.config.batch_size, self.config.y_dim, reuse=True)
        self.D_loss, self.G_loss = model_loss(logits_real, logits_fake)
        self.D_opt, self.G_opt = model_opt(self.D_loss, self.G_loss, self.config.lr, self.config.beta1)
    
    def train(self, sess, mnist, show_every=250, print_every=50):
        """Train a GAN for a certain number of epochs"""
        #WIP
        saver = tf.train.Saver()
        sample_z = np.random.uniform(-1, 1, size=(self.config.sample_size, self.config.z_dim))
        sample_y = convert_y_vec(np.random.randint(0, 10, size=self.config.sample_size))
        
        losses = []
        steps = 0
        
        start_epoch = 0 #load_checkpoint(sess, saver, self.config.checkpoint_dir)
        for e in range(start_epoch, self.config.num_epoch):
            max_iter = int(mnist.train.num_examples/self.config.batch_size)
            for _ in range(max_iter):
                minibatch, minibatch_y = mnist.train.next_batch(self.config.batch_size)
                minibatch_y = convert_y_vec(minibatch_y)
                steps += 1

                # Sample random noise for G
                batch_z = np.random.uniform(-1, 1, size=(self.config.batch_size, self.config.z_dim))
            
                # Run optimizers
                _, D_loss_curr = sess.run([self.D_opt, self.D_loss], feed_dict={self.input_real: minibatch, self.input_z: batch_z, self.input_y: minibatch_y})
                _, G_loss_curr = sess.run([self.G_opt, self.G_loss], feed_dict={self.input_z: batch_z, self.input_real: minibatch, self.input_y: minibatch_y})
                losses.append([D_loss_curr, G_loss_curr])

                # print loss every so often
                # We want to make sure D_loss doesn't go to 0
                if steps % print_every == 0:
                    print('... Epoch {}, Iter {}, D:{:.4}, G:{:.4}'.format(e, steps, D_loss_curr, G_loss_curr))

                # show sample result every so often 
                if steps % show_every == 1:
                    gen_samples = sess.run(generator(self.input_z, self.input_y, self.config.sample_size, self.config.y_dim, reuse=True, training=False),
                                           feed_dict={self.input_z: sample_z, self.input_y: sample_y})
                    _ = show_images(gen_samples)
                    plt.show()

            saver.save(sess, os.path.join(self.config.checkpoint_dir, 'GAN_epoch{}.ckpt'.format(e)))

        print('Final images')
        gen_samples = sess.run(generator(self.input_z, self.input_y, self.config.sample_size, self.config.y_dim, reuse=True, training=False), feed_dict={self.input_z: sample_z, self.input_y: sample_y})
        _ = show_images(gen_samples)
        plt.show()

        saver.save(sess, os.path.join(self.config.checkpoint_dir, 'GAN_final.ckpt'))
        return losses

def model_inputs(real_dim, z_dim, y_dim):
    inputs_real = tf.placeholder(tf.float32, (None, *real_dim), name='input_real')
    inputs_z = tf.placeholder(tf.float32, (None, z_dim), name='input_z')
    inputs_y = tf.placeholder(tf.float32, (None, y_dim), name='input_y')
    return inputs_real, inputs_z, inputs_y

def conv_cond_concat(x, y):
    """Concatenate conditioning vector on feature map axis."""
    x_shapes = x.get_shape()
    y_shapes = y.get_shape()
    return tf.concat([x, y*tf.ones([x_shapes[0], x_shapes[1], x_shapes[2], y_shapes[3]])], 3)

def discriminator(x, y, batch_size, y_dim, reuse=False):
    with tf.variable_scope("discriminator", reuse=reuse):
        init = tf.contrib.layers.xavier_initializer(uniform=True)

        x_img = tf.reshape(x, [batch_size,28,28,1])
        yb = tf.reshape(y, [batch_size, 1, 1, y_dim])
        x = conv_cond_concat(x_img, yb)
        
        h0 = tf.layers.conv2d(x, 1+y_dim, 5, strides=2, activation=leaky_relu, padding='same', kernel_initializer=init)
        h0 = conv_cond_concat(h0, yb)

        h1 = tf.layers.conv2d(h0, 64+y_dim, 5, strides=2, padding='same', kernel_initializer=init)
        h1 = tf.layers.batch_normalization(h1, training=True)
        h1 = leaky_relu(h1)
        h1 = tf.reshape(h1, [batch_size, -1])
        h1 = tf.concat([h1, y], 1)

        h2 = tf.layers.dense(h1, 1024, kernel_initializer=init)
        h2 = tf.layers.batch_normalization(h2, training=True)
        h2 = leaky_relu(h2)
        h2 = tf.concat([h2, y], 1)

        logits = tf.layers.dense(h2, 1, kernel_initializer=init)
        return logits

def generator(z, y, batch_size, y_dim, reuse=False, training=True):
    with tf.variable_scope("generator", reuse=reuse):
        init = tf.contrib.layers.xavier_initializer(uniform=True)

        yb = tf.reshape(y, [batch_size, 1, 1, y_dim])
        z = tf.concat([z, y], 1)

        h0 = tf.layers.dense(z, 1024, kernel_initializer=init)
        h0 = tf.layers.batch_normalization(h0, training=training)
        h0 = tf.nn.relu(h0)
        h0 = tf.concat([h0, y], 1)

        h1 = tf.layers.dense(h0, 128*7*7, kernel_initializer=init)
        h1 = tf.layers.batch_normalization(h1, training=training)
        h1 = tf.nn.relu(h1)
        
        h1 = tf.reshape(h1, [batch_size, 7, 7, 128])
        h1 = conv_cond_concat(h1, yb)

        h2 = tf.layers.conv2d_transpose(h1, 128, 5, strides=2, padding='same', kernel_initializer=init)
        h2 = tf.layers.batch_normalization(h2, training=training)
        h2 = tf.nn.relu(h2)
        h2 = conv_cond_concat(h2, yb)

        h3 = tf.layers.conv2d_transpose(h2, 1, 5, strides=2, padding='same', kernel_initializer=init)
        img = tf.nn.sigmoid(h3)
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
