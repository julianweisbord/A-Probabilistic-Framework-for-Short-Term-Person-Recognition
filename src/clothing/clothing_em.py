'''
Created on March 8th, 2018
author: Julian Weisbord
sources:
description:
'''

import sys
import random
import glob
import cv2

IMG_WIDTH = 299
IMG_HEIGHT = 299

class ClothingEM():

    def __init__(self, frame_dataset=None, video=False):
        self.images = frame_dataset[0]
        self.image_names = frame_dataset[1]
        self.person_locations = frame_dataset[2]
        self.video = video

    def add_frames(self, frames):
        for frame in frames:
            self.images.append(frame)

    def crop_to_person(self):
        pass

    def create_histogramss(self, img, line_height):
        # Crop image so that only the person is in the image:

        cv2.calcHist()

        return h1, h2

    def kmeans(self):
        '''
        K-means on physical space
        '''
        pass

    def color_em(self, img):
        # 0: Generate a random horizontal line (theta), classify points below and above it.
        rand_y = random.randint(1, IMG_HEIGHT + 1)
        random_line_img = cv2.line(img, (0, rand_y), (IMG_WIDTH, rand_y), (0, 0, 250), 4)
        # cv2.imshow('Image with random line', random_line)
        cv2.imwrite('../../manipulated_images/' + random_line_img)

        # 1: Build color histograms, Histogram1 (H1) is above the theta line,
            # and Histogram 2 (H2) is below. One histogram will likely contain data that
            # isn't like the rest of its data and is more like the other histogram data,
            # so the line must be moved
        self.create_histograms(random_line_img, rand_y)

		# 2: Reclassify Points, P(p | H1), P(p | H2) then relabel points

    def clothing_distribution(self):
        converged = None
        while not converged:
            self.color_em()
		# 3: (Physical Space) K-Means step on theta line, mean of the 2 distributions to
        self.kmeans()
