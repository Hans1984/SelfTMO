import time, math, os, sys, random
import tensorflow as tf
import tensorlayer as tl
import numpy as np
np.seterr(divide='ignore', invalid='ignore')
import tensorflow.contrib.slim as slim
sys.path.insert(0, "../")
import img_io
import network

flags = tf.app.flags

flags.DEFINE_string("data_dir", './data/', "hdr_path")
flags.DEFINE_string("log_dir", './output/log', "log_path")
flags.DEFINE_string("im_dir", "./output/results_test/", "im_path")
flags.DEFINE_string("check_dir_save", './output/ckpt/', "model_save_path")
flags.DEFINE_string("sx", '768', 'img_width')
flags.DEFINE_string("sy", '384', 'img_height')
flags.DEFINE_string("vgg_ckpt", "./vgg_ckpt/vgg_19.ckpt", "vgg_pretrained_model")

flags.DEFINE_float('learning_rate', 0.0002, 'the initial learning_rate')
flags.DEFINE_float('gamma', 0.5, 'the gamma')
flags.DEFINE_float('beta', 1.0, 'the beta')
flags.DEFINE_integer('max_epoch', 401, 'training epochs')
flags.DEFINE_integer('batch_size', 1, 'batch size')
flags.DEFINE_bool('is_vgg', True, 'use of VGG')
flags.DEFINE_integer('print_batch_freq', 400, 'frequency of print results')


FLAGS = tf.app.flags.FLAGS

data_dir = FLAGS.data_dir
log_dir = FLAGS.log_dir
im_dir = FLAGS.im_dir
check_dir_save = FLAGS.check_dir_save
sx = FLAGS.sx#512
sy = FLAGS.sy
vgg_ckpt = FLAGS.vgg_ckpt
gamma = FLAGS.gamma
beta = FLAGS.beta
learning_rate = FLAGS.learning_rate
max_epoch = FLAGS.max_epoch
batch_size = FLAGS.batch_size
is_vgg = FLAGS.is_vgg
print_batch_freq = FLAGS.print_batch_freq

print('gamma:\n', gamma)
print('beta:\n', beta)

tl.files.exists_or_mkdir(log_dir)
tl.files.exists_or_mkdir(check_dir_save)
tl.files.exists_or_mkdir(im_dir)

steps_per_epoch = 1

x = tf.placeholder('float32', shape=[batch_size, sy, sx, 3], name = 'hdr_ulaw')
y_ = tf.placeholder('float32', shape=[batch_size, sy, sx, 3], name = 'hdr_orig')

def lum(img):
    r = img[:,:,0]
    g = img[:,:,1]
    b = img[:,:,2]
    l = 0.2126 * r + 0.7152 * g + 0.0722 * b
    return l

def mul_exp(img):
    x_p = 1.21497
    c_start = tf.log((x_p/tf.reduce_max(img)))/tf.math.log(2.)
    c_end = tf.log(x_p/tf.contrib.distributions.percentile(img,50.))/tf.math.log(2.)
    output_list = []
    exp_value = [c_start, (c_end + c_start)/2.0, c_end]
    for i in range(len(exp_value)):
        sc = tf.pow(tf.sqrt(2.0), exp_value[i])
        img_exp = img*sc
        img_pow = img_exp
        img_out = tf.where(img_pow>1.0, tf.ones_like(img_pow), img_pow)
        output_list.append(img_out)
    return output_list

# Setup the network
print("Network setup:\n")
sess = tf.Session(config = tf.ConfigProto(allow_soft_placement = True, log_device_placement = False))

output_list = mul_exp(y_)
img_0_lum = output_list[0]
img_1_lum = output_list[1]
img_2_lum = output_list[2]

output = network.tmo_net(img_0_lum, img_1_lum, img_2_lum, name = 'tmo_net', is_training=True)

train_params_combined = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope = 'tmo_net')

for var in tf.trainable_variables():
       print(var.name)

cost_lum = network.FCM_loss(output, x, gamma = 0.5, beta = 1.0, sigma_num = 2, kernel_size_num = 13, kernel_size_den = 13)
cost_combined = cost_lum

global_step = tf.Variable(0, trainable=False)
starter_learning_rate = learning_rate

learning_rate = tf.train.exponential_decay(starter_learning_rate, global_step,
                                           10, 0.9, staircase=False)
extra_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)

with tf.control_dependencies(extra_update_ops):
    train_op_combined = tf.train.AdamOptimizer(learning_rate, beta1=0.9, beta2=0.999,
        epsilon=1e-8, use_locking=False).minimize(cost_combined, global_step=global_step, var_list = train_params_combined)


tf.summary.scalar("learning_rate", learning_rate)
summaries = tf.summary.merge_all()
file_writer = tf.summary.FileWriter(log_dir, sess.graph)
sess.run(tf.global_variables_initializer())

print("\nStarting training...\n")

if is_vgg:
    vgg_var_list = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='vgg_19')
    vgg_restore = tf.train.Saver(vgg_var_list)
    vgg_restore.restore(sess, vgg_ckpt)
    print('VGG19 restored successfully!!')

step = 0
k = 0
train_loss = 0.0
start_time = time.time()
start_time_tot = time.time()

var_list=tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='tmo_net')
saver = tf.train.Saver(var_list)

for ep in range(max_epoch):
    step += 1
    for name in os.listdir(data_dir):
        if os.path.splitext(name)[1] == '.exr':
            hdr_name = data_dir + '/' + name
            #print('name is', hdr_name)
            _, hdr_ulaw, hdr_rgb = img_io.load_hdr_ldr_norm_ulaw(hdr_name)   
            
            hdr_ulaw = hdr_ulaw[np.newaxis,:,:,:]
            hdr_rgb = hdr_rgb[np.newaxis,:,:,:]

            feed_dict = {x: hdr_ulaw, y_: hdr_rgb}
            _, err_t  = sess.run([train_op_combined, cost_combined], feed_dict)

            train_loss += err_t
            if step%10 == 0:
                saver.save(sess,os.path.join(check_dir_save,"ckpt_sep"), global_step)
            v = int(max(1.0, print_batch_freq/5.0))
            if (int(step) % v)  == 0:
                # Training and validation loss for Tensorboard
                train_summary = tf.Summary()
                train_summary.value.add(tag='training_loss',simple_value=train_loss/v)
                file_writer.add_summary(train_summary, step)

                # Other statistics for Tensorboard
                summary = sess.run(summaries)
                file_writer.add_summary(summary, step)
                file_writer.flush()
                print('  [Step %08d of %08d. Train loss = %0.8f]' % (step, max_epoch, train_loss/v))
                train_loss = 0.0


            if step % print_batch_freq == 0:

                hdr_ulaw_np, hdr_orig_np, y_predict = sess.run([x, y_, output], feed_dict)

                for i in range(0, x.shape[0]):
                    xx = np.squeeze(hdr_ulaw_np[i])
                    yy = np.squeeze(hdr_orig_np[i])
                    y_pred = np.squeeze(y_predict[i])

                    a = 0.6

                    r = yy[:,:,0]
                    g = yy[:,:,1]
                    b = yy[:,:,2]
                        
                    y_gt_lum_np = lum(yy)
                    yy_predict_np_lum = lum(y_pred)

                    img_out = np.zeros(np.shape(yy))
                    img_out[:, :, 0] = (r/y_gt_lum_np)**a*yy_predict_np_lum
                    img_out[:, :, 1] = (g/y_gt_lum_np)**a*yy_predict_np_lum
                    img_out[:, :, 2] = (b/y_gt_lum_np)**a*yy_predict_np_lum
                    
                    save_name = im_dir + name.replace('.exr', '.png')
                    img_io.writeLDR(img_out,  save_name)

                duration = time.time() - start_time
                duration_tot = time.time() - start_time_tot
                print('Timings:')
                print('       Since last: %.3f sec' % (duration))
                print('         Per step: %.3f sec' % (duration/print_batch_freq))
                print('        Per epoch: %.3f sec' % (duration*steps_per_epoch/print_batch_freq))
                print('')
                print('   Per step (avg): %.3f sec' % (duration_tot/step))
                print('  Per epoch (avg): %.3f sec' % (duration_tot*steps_per_epoch/step))
                print('')
                print('       Total time: %.3f sec' % (duration_tot))
                print('   Exp. time left: %.3f sec' % (duration_tot*steps_per_epoch*max_epoch/step - duration_tot))
                print('-------------------------------------------')
                start_time = time.time()


file_writer.close()
sess.close()