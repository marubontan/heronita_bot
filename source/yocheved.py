# -*- coding: utf-8 -*-
import requests
from io import BytesIO
import os
import time
import ConfigParser
from slackclient import SlackClient
from slacker import Slacker
from magenta.models.image_stylization import image_utils
import numpy as np
from scipy import misc
from style import Style

config = ConfigParser.SafeConfigParser()
config.read('./config.cfg')

BOT_ID = config.get('slack', 'bot_id')
BOT_NAME = config.get('slack', 'bot_name')
BOT_API = config.get('slack', 'bot_token')

# TODO: I couldn't post an image to slack on slackclient library.
slack_client = SlackClient(BOT_API)
slack = Slacker(BOT_API)


def handle_post(channel, img_url):
    # without header, It doesn't work
    res = requests.get(img_url, headers={'Authorization': 'Bearer %s' % BOT_API}, stream=True)

    # TODO: Now this doesn't use the image on memory.
    # TODO: Explore better resize.
    img_arr = misc.imresize(misc.imread(BytesIO(res.content), mode='RGB'), (200, 200))
    misc.imsave('../imgs/input/temp.jpg', img_arr)
    img = np.expand_dims(image_utils.load_np_image(
        os.path.expanduser('../imgs/input/temp.jpg')), 0)

    Style.download_checkpoints('check_points')

    style = Style()
    generated_imgs = style.generate_image(img)

    for i, generated_img in enumerate(generated_imgs):
        # TODO: Now this doesn't use the image on memory.
        file_name = '../imgs/output/' + 'generated_' + str(i) + '.jpg'
        misc.imsave(file_name, generated_img)

        slack.files.upload(file_name, filename=file_name, channels=channel)


def parse_slack_output(slack_output):
    if len(slack_output) > 0:
        output = slack_output[0]
        if 'text' in output:
            text = output['text'].split(' ')
            if BOT_NAME == text[len(text) - 1]:
                return output['channel'], output['file']['url_private_download']

    return None, None


if __name__ == "__main__":
    READ_WEB_SOCKET_DELAY = 1
    if slack_client.rtm_connect():
        while True:
            channel, img_url = parse_slack_output(slack_client.rtm_read())
            if channel and img_url:
                handle_post(channel, img_url)
            time.sleep(READ_WEB_SOCKET_DELAY)
