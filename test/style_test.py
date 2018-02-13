import sys

sys.path.append('../source')
from style import Style
import matplotlib.pyplot as plt
from magenta.models.image_stylization import image_utils
import numpy as np
import os

if __name__ == '__main__':
    # load image
    img = np.expand_dims(image_utils.load_np_image(
        os.path.expanduser('../sample/test.jpg')), 0)

    Style.download_checkpoints('check_points')

    style = Style()
    generated_imgs = style.generate_image(img)

    for i,generated_img in enumerate(generated_imgs):
        file_name = 'style_' + str(i) + '.jpg'
        plt.imsave(file_name, generated_img)
