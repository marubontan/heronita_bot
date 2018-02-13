import os
import random
import urllib2
import tensorflow as tf
from magenta.models.image_stylization import model


class Style():
    def __init__(self):
        self.original_image = None
        self.style = 'varied'
        self.checkpoints = 'check_points/multistyle-pastiche-generator-varied.ckpt'
        self.num_styles = 32

    def change_style(self, style):
        self.style = style
        if self.style == 'monet':
            self.checkpoints = 'check_points/multistyle-pastiche-generator-monet.ckpt'
            self.num_styles = 10

    def generate_image(self, image):
        styles = range(self.num_styles)
        random.shuffle(styles)
        which_styles = styles[0:6]

        with tf.Graph().as_default(), tf.Session() as sess:
            stylized_images = model.transform(tf.concat([image for _ in range(len(which_styles))], 0),
                                              normalizer_params={'labels': tf.constant(which_styles),
                                                                 'num_categories': self.num_styles, 'center': True,
                                                                 'scale': True})

            model_saver = tf.train.Saver(tf.global_variables())
            model_saver.restore(sess, self.checkpoints)
            stylized_images = stylized_images.eval()

            return stylized_images

    @staticmethod
    def download_checkpoints(checkpoint_dir):
        """
        This method makes a directory and downloads checkpoints data to that.
        """

        # check directory
        if not os.path.isdir(checkpoint_dir):
            os.mkdir(checkpoint_dir)

        url_prefix = 'http://download.magenta.tensorflow.org/models/'
        checkpoints = ['multistyle-pastiche-generator-monet.ckpt', 'multistyle-pastiche-generator-varied.ckpt']
        for checkpoint in checkpoints:
            full_checkpoint = os.path.join(checkpoint_dir, checkpoint)
            if not os.path.exists(full_checkpoint):
                print 'Downloading', full_checkpoint
                response = urllib2.urlopen(url_prefix + checkpoint)
                data = response.read()
                with open(full_checkpoint, 'wb') as fh:
                    fh.write(data)
