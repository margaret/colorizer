#!/usr/bin/env python
# -*- coding: utf-8 -*-
# CS 194-26 Project 1 starter code translated to Python 2.7
# https://inst.eecs.berkeley.edu/~cs194-26/fa14/hw/proj1/index.html

import sys
import numpy as np
from skimage import io, util
import matplotlib.pyplot as plt
from scipy.misc import imread #, imfilter, imresize, imrotate, imsave, imshow


def colorize(bw_file):
    """
    :param bw_file:
    the str filename or file handle of the grayscale image to be colorized.
    The file is 3 filtered copies of the target image vertically stacked in
    BGR order.

    :return np.ndarray color_img:
    colorized image
    """

    full_img = imread(bw_file)
    height = len(full_img)/3
    b, g, r = np.split(full_img, [height, height*2])
    aligned_g = util.img_as_float(align_imgs(g, b))
    aligned_r = util.img_as_float(align_imgs(r, b))
    b = util.img_as_float(b)
    print aligned_r.shape
    print aligned_g.shape
    print b.shape
    return np.dstack((aligned_r,aligned_g,b))


def align_imgs(img1, img2):
    """
    params img1 and img2 are ndarrays
    returns img1 aligned to img2? not sure what this is supposed to
    do but for now it just returns img1 arbitrarily cropped to match
    the shape of img2. Only could differ on height, since the original
    images were concatenated in a vertical stack.
    """
    aligned = None
    if img1.shape == img2.shape:
        return img1
    else:
        # img1 is smaller than img2
        difference = img1.shape[0] - img2.shape[0]
        if difference > 0:
            aligned = util.crop(img1, ((0,difference), (0,0)))
        else:
            aligned = np.append(img1, np.zeros(shape=(-difference, img1.shape[1])), axis=0)
        return aligned



def show_ndarr_as_img(img):
    # I would have just used scipy.misc.imshow except
    # this line: cmd = os.environ.get('SCIPY_PIL_IMAGE_VIEWER', 'see')
    # in https://github.com/scipy/scipy/blob/master/scipy/misc/pilutil.py
    # can't find 'see' and it's impossible to google what that's supposed
    # to do. I think working with ndarray's will be better for most of
    # the computational stuff so that's why I'm not just using PIL :\
    # fuck this shit.
    """
    param img: is an ndarray or filename
    returns: None
    """
    plt.figure()
    if type(img) == file or type(img) == str:
        img = io.imread(img)
    io.imshow(img)
    plt.show()
    plt.clf()

if __name__ == "__main__":
    try:
        bw_filename = sys.argv[1]
    except Error as e:
        print "Usage:\n$ python colorize.py imagename.ext"
        print e

    color_img = colorize(bw_filename)
    print color_img.shape
    show_ndarr_as_img(color_img)


