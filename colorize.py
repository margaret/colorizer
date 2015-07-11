#!/usr/bin/env python
# -*- coding: utf-8 -*-
# CS 194-26 Project 1 starter code translated to Python 2.7
# https://inst.eecs.berkeley.edu/~cs194-26/fa14/hw/proj1/index.html

import sys
import numpy as np
from skimage import io, util
from sklearn.metrics.pairwise import pairwise_distances
from itertools import product
import matplotlib.pyplot as plt
from scipy.misc import imread, imsave #, imfilter, imresize, imrotate, imshow


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
    b, g, r = map(util.img_as_float, np.split(full_img, [height, height*2]))
    r = align_imgs(r, g)
    print b.shape
    print g.shape
    print r.shape
    aligned_r, aligned_b = align_naive(r,b)
    aligned_g, aligned_b = align_naive(g,b)
    # aligned_g = util.crop(aligned_g, ((0,3),(0,1)))
    # aligned_b = util.crop(aligned_b, ((0,3),(0,1)))
    # aligned_r = align_imgs(r,b)
    # aligned_g = align_imgs(g,b)
    # aligned_b = b
    print aligned_r.shape
    print aligned_g.shape
    print aligned_b.shape
    return np.dstack((aligned_g,aligned_g,aligned_b))


def align_imgs(img1, img2):
    """
    more like make img1 same height as img2
    params img1 and img2 are ndarrays
    returns img1 cropped on the bottom to match
    the shape of img2. Only could differ on height, since the original
    images were concatenated in a vertical stack.

    ok this is
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
            # aligned = np.append(img1, np.zeros(shape=(-difference, img1.shape[1])), axis=0)
            aligned = util.pad(img1, ((0,difference), (0,0)), mode='edge')
        return aligned


def align_naive(img1, img2, window=15, scorer='ncc'):
    """
    find optimal alignment if of img1 wrt img2
    :param img1: ndarray
    :param img2: ndarray
    :return:
    """
    def ncc(img1,img2):
        """
        normalized cross-correlation of images
        dot product of normalized vectors

        :param img1:
        :param img2:
        :return:
        """
        normed1 = normalize(img1).flatten()
        normed2 = normalize(img2).flatten()
        return normed1.dot(normed2)

    metrics = {'ncc':ncc}

    distance = metrics[scorer]
    best = 0
    aligned1 = None
    aligned2 = None
    for displacement in product(xrange(window), xrange(window)):
        # print displacement
        disp1, disp2 = displace(img1, img2, displacement)
        score = distance(disp1, disp2)
        # score = pairwise_distances(disp1, disp2, metric='cosine')
        if score > best:
            print "new score", score, displacement
            best = distance(disp1, disp2)
            aligned1, aligned2 = disp1, disp2
    print best
    return aligned1, aligned2


def displace(img1, img2, d):
    """
    :param img1:
    :param img2: shapes should be equal
    :param displacement: tuple (x,y) to shift img2 wrt img1 where positive
     numbers indicate down/right and negative displacement is up/left
    :return img1, img2: cropped/displaced
    examples
    img1    img2
    1 2 3   a b c
    4 5 6   d e f
    7 8 9   g h i

    (1,0) returns     (-1, 1) returns
    1 2 3   d e f       4 5    b c
    4 5 6   g h i       7 8    e f
    """
    x,y = d
    disp = lambda x: (0,x) if x > 0 else (-x,0)
    disp1 = util.crop(img1, (disp(x), disp(y)))
    disp2 = util.crop(img2, (disp(-x), disp(-y)))
    return disp1, disp2


def normalize(a):
    # assumes floats
    row_sums = a.sum(axis=1)
    return a / row_sums[:, np.newaxis]


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
        color_filename = sys.argv[2]
    except Exception as e:
        print "Usage:\n$ python colorize.py imagename.ext"
        print e

    color_img = colorize(bw_filename)
    print color_img.shape

    show_ndarr_as_img(color_img)

    if len(sys.argv) > 2:
        color_filename = sys.argv[2]
        io.imsave(color_filename, color_img)

