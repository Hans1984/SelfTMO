import os, sys
#os.environ['CUDA_VISIBLE_DEVICES'] = "0"
import tensorflow as tf
import tensorlayer as tl
import numpy as np
np.seterr(divide='ignore', invalid='ignore')
import network, img_io
import time

eps = 1/255.0

def print_(str, color='', bold=False):
    if color == 'w':
        sys.stdout.write('\033[93m')
    elif color == "e":
        sys.stdout.write('\033[91m')
    elif color == "m":
        sys.stdout.write('\033[95m')

    if bold:
        sys.stdout.write('\033[1m')

    sys.stdout.write(str)
    sys.stdout.write('\033[0m')
    sys.stdout.flush()

# Settings, using TensorFlow arguments
flags = tf.app.flags
flags.DEFINE_string("im_dir", "./data_offline/", "Path to image directory or an individual image")
flags.DEFINE_string("out_dir", "./output/results_offline/", "Path to output directory")
flags.DEFINE_string("check_dir", "./output/offline_model/ckpt/", "Path to check_checkpoint directory")
flags.DEFINE_string("sx", "768", "Reconstruction image width")
flags.DEFINE_string("sy", "384", "Reconstruction image height")
# flags.DEFINE_string("img_name", '', "im_name_save")

FLAGS = tf.app.flags.FLAGS

sx = FLAGS.sx
sy = FLAGS.sy
check_dir = FLAGS.check_dir

if os.path.isdir(FLAGS.im_dir):

    frames = [os.path.join(FLAGS.im_dir, name)
              for name in sorted(os.listdir(FLAGS.im_dir))
              if os.path.isfile(os.path.join(FLAGS.im_dir, name))]

# Placeholder for image input
x = tf.placeholder('float32', shape=[1, sy, sx, 3])

print_("Network setup:\n")

def lum_np(img):
    r = img[:,:,0]
    g = img[:,:,1]
    b = img[:,:,2]
    l = 0.2126 * r + 0.7152 * g + 0.0722 * b #0.299*r + 0.587*g + 0.114*b
    return l

def lum2rgb(img_gt, img_gt_lum, img_out_lum, a = 0.6):
    a = 0.6

    r = img_gt[:,:,0]
    g = img_gt[:,:,1]
    b = img_gt[:,:,2]

    img_out = np.zeros(np.shape(img_gt))
    img_out[:, :, 0] = (r/img_gt_lum)**a*img_out_lum
    img_out[:, :, 1] = (g/img_gt_lum)**a*img_out_lum
    img_out[:, :, 2] = (b/img_gt_lum)**a*img_out_lum

    return img_out

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

output_list = mul_exp(x)
img_0_lum = output_list[0]
img_1_lum = output_list[1]
img_2_lum = output_list[2]
img_gt_lum = x

out = network.tmo_net(img_0_lum, img_1_lum, img_2_lum, name = 'tmo_net')

saver = tf.train.Saver(tf.global_variables())
sess = tf.InteractiveSession()

var_list_restore = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='tmo_net')
restore_saver = tf.train.Saver(var_list_restore)
chkpt_fname = tf.train.latest_checkpoint(check_dir)
restore_saver.restore(sess, chkpt_fname)

print_("\tdone\n")

if not os.path.exists(FLAGS.out_dir):
    os.makedirs(FLAGS.out_dir)

print_("\nStarting prediction...\n\n")
k = 0
j = 0;
start = time.time()
for i in range(len(frames)):
    print("Frame %d: '%s'"%(i,frames[i]))

    try:
        # Read frame
        print_("\tReading...")
        _, _, x_buffer_np = img_io.load_hdr_ldr_norm_ulaw(frames[i])
        x_buffer = x_buffer_np[np.newaxis,:,:,:]
        print_("\tdone")
        print_("\tInference...")
        feed_dict = {x: x_buffer}
        y_predict = sess.run([out], feed_dict=feed_dict)
        y_predict_sq = np.squeeze(y_predict)
        y_predict_lum = lum_np(y_predict_sq)
        y_gt_lum = lum_np(x_buffer_np)
        out_img = lum2rgb(x_buffer_np, y_gt_lum, y_predict_lum)

        print_("\tdone\n")

        print_("\tWriting...")
        k += 1;
        (file_path, temp_name) = os.path.split(frames[i])
        print(os.path.join(FLAGS.out_dir, temp_name.replace('.exr', '.png')))
        img_io.writeLDR(out_img, os.path.join(FLAGS.out_dir, temp_name.replace('.exr', '.png')))
        print_("\tdone\n")

    except img_io.IOException as e:
        print_("\n\t\tWarning! ", 'w', True)
        print_("%s\n"%e, 'w')
    except Exception as e:    
        print_("\n\t\tError: ", 'e', True)
        print_("%s\n"%e, 'e')

end = time.time()
print('total time is :', (end - start))
print('time per frames is :', (end- start)/len(frames))

print_("Done!\n")

sess.close()
