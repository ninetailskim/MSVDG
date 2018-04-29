import argparse
import tensorflow as tf
from watch_data import get_data as Decoderlmdb  # noqa
from tensorpack import *
from tensorpack.tfutils.summary import add_moving_summary
from tensorpack.utils import logger
from tensorpack.tfutils.scope_utils import auto_reuse_variable_scope
import tensorpack.tfutils.symbolic_functions as symbf
import glob
import os
from GAN import GANTrainer, GANModelDesc

SEQ_LEN = 5
BATCH = 1
IN_CH = 3
OUT_CH = 3
LAMBDA = 100
LEVELS = 3
SHAPE = 512
NF = 64

def ReluConv2D(name, x, out_channels, use_relu=True, kernel_shape=3, stride=1):
    if use_relu:
        x = tf.nn.relu(x, name='%s_relu' % name)
    x = Conv2D('%s_conv' % name, x, out_channels, kernel_shape=kernel_shape, stride=stride)
    return x

def ReluDeconv2D(name, x, out_channels, kernel_shape=3, stride=1):
    x = tf.nn.relu(x, name='%s_relu' % name)
    x = Deconv2D('%s_deconv' % name, x, out_channels, kernel_shape=kernel_shape, stride=stride)
    return x

def resize(small, name):
    with tf.variable_scope('resize'):
        out_channels = small.get_shape().as_list()[3]
        small = Deconv2D('spatial_skip_deconv_%s' % name, small, out_channels, kernel_shape=4, stride=2)
        small = tf.nn.relu(small, name='spatial_skip_relu_%s' % name)
        return small

def resize_by_factor(x, f):
    with tf.name_scope('resize'):
        height, width = x.get_shape().as_list()[1:3]
        return tf.image.resize_images(x, [int(height * f), int(width * f)])

def Merge(incoming_skip, ID, tensor, name):
    with tf.name_scope('Merge_%s' % name):
        if incoming_skip is None:
            incoming_skip_internel = tensor
        else:
            incoming_skip_internel = incoming_skip[ID]
        #get size, origin size should be [batch, height, weight, channel]
        hs, ws = incoming_skip_internel.get_shape().as_list()[1:3]
        hl, wl = tensor.get_shape().as_list()[1:3]

        tmp_name = resize(incoming_skip_internel, name)
        if(hs != hl) or (ws != wl):
            incoming_skip_internel = tmp_name
        channels = tensor.get_shape().as_list()[3]
        tensor_internal = tf.concat([tensor, incoming_skip_internel], axis=3)
        tensor_internal = ReluConv2D(name, tensor_internal, channels, kernel_shape=1)
        
        if incoming_skip is None:
            return tensor
        else:
            return tensor_internal


def BNLReLU(x, name=None):
    x = BatchNorm('bn',x)
    return tf.nn.leaky_relu(x,alpha=0.2, name=name)

class Model(GANModelDesc):
    def inputs(self):
        return [tf.placeholder(tf.float32, (None, SEQ_LEN, SHAPE, SHAPE, 3), 'blurry'),
                tf.placeholder(tf.float32, (None, SEQ_LEN, SHAPE, SHAPE, 3), 'sharp')]

    @auto_reuse_variable_scope
    def generator(self, observation, estimate, skip_temporal_in=None, skip_spatial_in=None, name=None):

        skip_temporal_out = []
        skip_spatial_out = []
        skip_unet_out = []

        with tf.name_scope("deblur_block_%s" % name):

            with argscope(BatchNorm, training=True), argscope([Conv2D, Deconv2D], activation=BNLReLU):
                inputs = tf.concat([observation, estimate], 3)

                block = ReluConv2D('d0', inputs, 32, stride=1, kernel_shape=3)

                with tf.name_scope('block_0'):
                    block = ReluConv2D('d1_0', block, 64, stride=2)
                    block_start = block
                    block = Merge(skip_spatial_in, 0, block, 'd14_s')
                    block = ReluConv2D('d1_1', block, 64)
                    block = ReluConv2D('d1_2', block, 64)
                    block = ReluConv2D('d1_3', block, 64, kernel_shape=1)
                    block = tf.add(block_start, block, name='block_skip_A')
                    skip_spatial_out.append(block)

                
                with tf.name_scope('block_1'):
                    block = ReluConv2D('d2_0', block, 64)
                    block_start = block
                    block = Merge(skip_spatial_in, 1, block, 'd24_s')
                    block = ReluConv2D('d2_1', block, 64)
                    block = ReluConv2D('d2_2', block, 64)
                    block = ReluConv2D('d2_3', block, 64, kernel_shape=1)
                    block = tf.add(block_start, block, name='block_skip_A')
                    skip_spatial_out.append(block)
                    skip_unet_out.append(block)


                with tf.name_scope('block_2'):
                    block = ReluConv2D('d3_0', block, 128, stride=2)
                    block = Merge(skip_spatial_in, 2, block, 'd34_s')
                    block_start = block
                    block = ReluConv2D('d3_1', block, 128)
                    block = ReluConv2D('d3_2', block, 128)
                    block = ReluConv2D('d3_3', block, 128, kernel_shape=1)
                    block = tf.add(block_start,block, name='block_skip_c')
                    skip_spatial_out.append(block)
                    skip_unet_out.append(block)

                with tf.name_scope('block_3'):
                    block = ReluConv2D('d4_0', block, 256, stride=2)
                    block = Merge(skip_spatial_in, 3, block, 'd33_s')
                    block_start = block
                    block = Merge(skip_temporal_in, 0, block, 'd41_s')
                    block = ReluConv2D('d4_1', block, 256)
                    block = ReluConv2D('d4_2', block, 256)
                    block = ReluConv2D('d4_3', block, 256, kernel_shape=1)
                    block = tf.add(block_start,block, name='block_skip_D')
                    skip_temporal_out.append(block)
                    skip_spatial_out.append(block)

                with tf.name_scope('block_4'):
                    block = ReluDeconv2D('u1_0', block, 128, stride=2, kernel_shape=4)
                    block = tf.add(block, skip_unet_out[1], name='skip01')
                    block_start = block
                    block = Merge(skip_temporal_in, 1, block, 'u1_s')
                    block = ReluConv2D('u1_1', block, 128)
                    block = ReluConv2D('u1_2', block, 128)
                    block = ReluConv2D('u1_3', block, 128)
                    block = tf.add(block, block_start, name='block_skip_E')
                    skip_temporal_out.append(block)

                
                with tf.name_scope('block_5'):
                    block = ReluDeconv2D('u2_0', block, 64, stride=2, kernel_shape=4)
                    block = tf.add(block, skip_unet_out[0], name='skip01')
                    block_start = block
                    block = Merge(skip_temporal_in, 2, block, 'u2_s')
                    block = ReluConv2D('u2_1', block, 64)
                    block = ReluConv2D('u2_2', block, 64)
                    block = ReluConv2D('u2_3', block, 64)
                    block = tf.add(block, block_start, name='block_skio_F')
                    skip_temporal_out.append(block)

                with tf.name_scope('block_6'):
                    block = ReluDeconv2D('u3_0', block, 64, stride=2, kernel_shape=4)
                    block = ReluConv2D('u3_1', block, 64)
                    block = ReluConv2D('u3_2', block, 64)
                    block = ReluConv2D('u3_3', block, 66)
                    block = ReluConv2D('u3_4', block, 3)

                estimate = tf.add(estimate, block, name='skip03')

                return estimate, skip_spatial_out, skip_temporal_out
    
    @auto_reuse_variable_scope
    def discriminator(self, inputs, outputs, lev):
        #process input here
        l = tf.concat([inputs, outputs],3)
        with argscope(Conv2D, kernel_size=4, strides=2, activation=BNReLU):
            if lev == 0:
                l = (LinearWrap(l)
                    .Conv2D('conv0_%d' %lev, NF, activation=tf.nn.leaky_relu)
                    .Conv2D('conv1_%d' %lev, NF * 2, strides=1)
                    .Conv2D('convlast_%d' %lev, 1, strides=1, activation=tf.identity)())
            else:
                if lev == 1:
                    l = (LinearWrap(l)
                        .Conv2D('conv0', NF, activation=tf.nn.leaky_relu)
                        .Conv2D('conv1', NF * 2)
                        .Conv2D('conv2', NF * 4, strides=1)
                        .Conv2D('convlast', 1, strides=1, activation=tf.identity)())
                else:
                    if lev == 2:
                        l = (LinearWrap(l)
                            .Conv2D('conv0', NF, activation=tf.nn.leaky_relu)
                            .Conv2D('conv1', NF * 2)
                            .Conv2D('conv2', NF * 4)
                            .Conv2D('conv3', NF * 8, strides=1)
                            .Conv2D('convlast', 1, strides=1, activation=tf.identity)())
            return l 

    def build_graph(self, blurry, sharp):

        def l2_loss(x, y, name):
            return tf.reduce_mean(tf.squared_difference(x, y), name=name)

        def l1_loss(x, y, name):
            return tf.reduce_mean(tf.abs(x - y), name=name)

        def scaled_psnr(x, y, name):
            return symbf.psnr(128. * (x + 1.0), 128.0 * (y + 1.), 255, name=name)
        
        #return from small to big
        def image_pyrmaid(img, levels=LEVELS):
            with tf.name_scope('image_pyramid'):
                pyramid = []
                for i in range(levels):
                    pyramid.append(resize_by_factor(img, 1. / (2 **(i + 1))))
            return pyramid[::-1]
        #input 1
        #input, output = input / 128.0 - 1, output / 128.0 - 1
        #input 2
        #blurry, sharp = input_vars

        blurry = blurry / 128.0 - 1
        sharp = sharp / 128.0 - 1

        #last sharp
        expected_pyramid = image_pyrmaid(sharp[:, -1, :, :, :], levels=LEVELS)
        #last blur
        estimate_pyramid = image_pyrmaid(blurry[:, -1, :, :, :], levels=LEVELS)
        '''
        l2err_list, l1err_list, psnr_list, psnr_impro_list = [], [], [], []
        for _ in range(LEVELS):
            l2err_list.append([])
            l1err_list.append([])
            psnr_list.append([])
            psnr_impro_list.append([])
        '''
        cost_list = []
        l2err_list = [ [None] * 5 for i in range(3) ]
        l1err_list = [ [None] * 5 for i in range(3) ]
        psnr_list = [ [None] * 5 for i in range(3) ]
        psnr_impro_list = [ [None] * 5 for i in range(3) ]
        skip_spatial_out = [None] * LEVELS
        skip_temporal_out  = [None] * LEVELS

        estimate_viz = []

        #also the last blur
        baseline_pyramid = image_pyrmaid(blurry[:, SEQ_LEN - 1, :, :, :], levels=LEVELS)
        #calculate the origin PSNR
        psnr_base = [scaled_psnr(x, y, name="PSNR_base") for x, y in zip(baseline_pyramid, expected_pyramid)]

        for t in range(1, SEQ_LEN):
            logger.info("build time step: %i" % t)
            #1-3
            #2-2
            #3-1
            #4-0
            observation_pyramid = image_pyrmaid(blurry[:, SEQ_LEN - t - 1, :, :, :], levels=LEVELS)
            
            for l in range(LEVELS):
                #l      l1
                #0      2
                #1      1
                #2      0
                ll = LEVELS - l - 1
                logger.info("level:{} with input shape {}".format(ll, observation_pyramid[l].get_shape()))
                skip_spatial_in = None if (l == 0) else skip_spatial_out[l - 1]
                with tf.variable_scope('gen_%i' %l):
                    estimate_pyramid[l], skip_spatial_out[l], skip_temporal_out[l] = \
                        self.generator(observation_pyramid[l], 
                                        estimate_pyramid[l], 
                                        skip_temporal_in=skip_temporal_out[l],
                                        skip_spatial_in=skip_spatial_in,
                                        name = 'level_%i_step_%i' %(ll, t))
                with argscope([Conv2D, Conv2DTranspose], kernel_initializer=tf.truncated_normal_initializer(stddev=0.02)):
                    with tf.variable_scope('discrim_%i' %l):
                        real_pred = self.discriminator(observation_pyramid[l], expected_pyramid[l], l)
                        fake_pred = self.discriminator(observation_pyramid[l], estimate_pyramid[l], l)

                self.build_losses(real_pred, fake_pred, l)
                errL1 = tf.reduce_mean(tf.abs(estimate_pyramid[l] - expected_pyramid[l]), name='L1_loss_%d' %l)
                self.my_g_loss[l] = tf.add(self.my_g_loss[l], LAMBDA * errL1, name='total_g_loss_%d' %l)
                add_moving_summary(errL1, self.my_g_loss[l])

                l2err_list[l][t] = l2_loss(estimate_pyramid[l], expected_pyramid[l], name="L2loss_t%i_l%i" % (t, ll))
                l1err_list[l][t] = l1_loss(estimate_pyramid[l], expected_pyramid[l], name="L1loss_t%i_l%i" % (t, ll))
                psnr_list[l][t] = scaled_psnr(estimate_pyramid[l], expected_pyramid[l], name="PSNR_t%i_l%i" % (t, ll))
                pi = tf.divide(psnr_list[l][t], psnr_base[l], name="PSNR_IMPRO_t%i_l%i" % (t, ll))
                psnr_impro_list[l][t] = pi

                #if ll == 0:
                    #cost_list.append(l2err_list[l][-1])

                tf.identity((estimate_pyramid[l] + 1.0) * 128., name='estimate_t%i_l%i' % (t, ll))

                if(l == LEVELS - 1):
                    estimate_viz.append(estimate_pyramid[l])

        with tf.name_scope('visualization'):
            expected = resize_by_factor(sharp[:, -1, :, :, :], 0.5)
            estimate_viz = tf.concat(estimate_viz, axis=2)
            observed = tf.concat([resize_by_factor(blurry[:, i, :, :, :], 0.5) for i in range(SEQ_LEN)], axis=2)

            viz = tf.concat([observed, estimate_viz, expected], axis=2, name='estimates')
            viz = 128.0 * (viz + 1.0)
            viz = tf.cast(tf.clip_by_value(viz, 0, 255), tf.uint8, name='viz')
            tf.summary.image('blurry5_estimates5_expected', viz, max_outputs=max(1, BATCH))
        ###############################################################################

        #if IN_CH == 1:
        #    input = tf.image.grayscale_to_rgb(input)
        #if OUT_CH == 1:
        #    output = tf.image.grayscale_to_rgb(output)
        #    fake_output = tf.image.grayscale_to_rgb(fake_output)
        ###############################################################################
        #self.cost = tf.add_n(cost_list, name="total_cost")
        #add_moving_summary(self.cost)

        for l in range(LEVELS):
            for t in range(1,SEQ_LEN):
                add_moving_summary(psnr_list[l][t], psnr_impro_list[l][t])

        self.collect_variables()

    def optimizer(self):
        lr = tf.get_variable('learning_rate', initializer=2e-4, trainable=False)
        return tf.train.AdamOptimizer(lr, beta1=0.5, epsilon=1e-3)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--datadir', help='data', required=True)
    parser.add_argument('--load', help='load model')
    parser.add_argument('--batch', help="batch_size", type=int, default=1)
    parser.add_argument('--gpu', help='gpu list', default=0)
    parser.add_argument('--train', help='mode', default=False)
    parser.add_argument('--logname',help="modify the log dir")
    args = parser.parse_args()

    if args.gpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    BATCH = args.batch

    print args.datadir
    print os.path.join(args.datadir, '*.lmdb?')
    if args.train:
        #auto_set_dir(action=None, name=None)
        logger.auto_set_dir(name=args.logname)
        #how to get data

        lmdbs = glob.glob(os.path.join(args.datadir, '*.lmdb?'))
        ds_train = [Decoderlmdb(lmdb) for lmdb in lmdbs]
        ds_train = RandomMixData(ds_train)
        ds_train = BatchData(ds_train, BATCH)
        ds_train = PrefetchData(ds_train, 100, 1)

        #lmdbs = glob.glob(os.path.join(datadir, 'val*.lmdb'))
        #ds_val = [Decoderlmdb(lmdb) for lmdb in lmdbs]
        #ds_val = RandomMixData(ds_val)
        #ds_val = BatchData(ds_val, BATCH_SIZE)
        #ds_val = FixedSizeData(ds_val, 100)
        #ds_val = PrefetchDataZMQ(ds_val, 8)
        
        data = QueueInput(ds_train)

        GANTrainer(data, Model()).train_with_defaults(
            callbacks=[
                PeriodicTrigger(ModelSaver(), every_k_epochs=3),
                ScheduledHyperParamSetter('learning_rate',[(200, 1e-4)])
            ],
            steps_per_epoch=data.size(),
            max_epoch=300,
            session_init=SaverRestore(args.load) if args.load else None
        )

    