'''
Created on March 8th, 2018
author: Julian Weisbord
sources: https://docs.opencv.org/3.1.0/d1/db7/tutorial_py_histogram_begins.html
description:
'''

import sys
import random
import glob
import numpy as np
import cv2
from matplotlib import pyplot as plt

IMG_WIDTH = 299
IMG_HEIGHT = 299
VERBOSITY = 2

class ClothingEM():

    def __init__(self, frame_dataset=None, video=False):
        self.images = frame_dataset[0]
        self.image_names = frame_dataset[1]
        self.person_locations = frame_dataset[2]
        if video:
            self.video = video
            # Convert video to a series of frames

    def add_frames(self, dataset):
        frames = []
        names = []
        locations = []
        for person_data in dataset:
            self.images.append(dataset[0])
            self.image_names.append(dataset[1])
            self.person_locations.append(dataset[0])

    def crop_to_person(self):
        # Simulate this function for now with constant values
        x1 = 89
        x2 = 198
        y1 = 0
        y2 = 284
        return x1, x2, y1, y2

    def create_histograms(self, img, line_height):
        if VERBOSITY >=2:
            l = cv2.imshow('Random Line Splitting Clothing', img)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
            print("img shape: ", img.shape)
        # Crop image so that only the person is in the image:
        x1, x2, y1, y2 = self.crop_to_person()

        mask_upper_cropped = np.zeros(img.shape[:2], np.uint8)
        mask_upper_cropped[y1:line_height, x1:x2] = 255
        upper_masked_img_cropped = cv2.bitwise_and(img, img, mask=mask_upper_cropped)

        # mask_upper = np.zeros(img.shape[:2], np.uint8)
        # # mask_upper[<height>, <width>]
        # mask_upper[line_height:img.shape[1], 0:img.shape[0]] = 255
        # upper_masked_img = cv2.bitwise_and(img, img, mask=mask_upper)

        # mask_lower = np.zeros(img.shape[:2], np.uint8)
        # mask_lower[0:line_height, 0:img.shape[1]] = 255
        # lower_masked_img = cv2.bitwise_and(img, img, mask = mask_lower)

        mask_lower_cropped = np.zeros(img.shape[:2], np.uint8)
        mask_lower_cropped[line_height:y2, x1:x2] = 255
        lower_masked_img_cropped = cv2.bitwise_and(img, img, mask=mask_lower_cropped)



        # Create upper and lower histograms and plot them
        plt.figure("Color Histograms")
        upper_plot = plt.subplot(221)
        upper_plot.set_title("Above Theta Line")
        lower_plot = plt.subplot(222)
        lower_plot.set_title("Below Theta Line")
        if VERBOSITY >=2:
            upper_plot.imshow(upper_masked_img_cropped)
            lower_plot.imshow(lower_masked_img_cropped)

        colors = ("b", "g", "r")
        features_h1 = []
        features_h2 = []
        for i,col in enumerate(colors):
            h1 = cv2.calcHist([img], [i], mask_upper_cropped, [256], [0, 1])
            h2 = cv2.calcHist([img], [i], mask_lower_cropped, [256], [0, 1])
            features_h1.extend(h1)
            features_h2.extend(h2)
            h1_plot = plt.subplot(223)
            h1_plot.set_title("Histogram Above Theta Line")
            h1_plot.plot(h1, color=col)
            plt.xlim([0, 256])
            h2_plot = plt.subplot(224)
            h2_plot.set_title("Histogram Below Theta Line")
            h2_plot.plot(h2, color=col)
            plt.xlim([0, 256])
        plt.show()
        print("Flattened feature vector", np.array(features_h1).flatten().shape)
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

        # Save random line image
        random_line_img_out = np.multiply(random_line_img, 255.0)
        random_line_img_out = random_line_img_out.astype('uint8')
        cv2.imwrite('../../data_capture/manipulated_images/line.jpeg', random_line_img_out)

        # 1: Build color histograms, Histogram1 (H1) is above the theta line,
            # and Histogram 2 (H2) is below. One histogram will likely contain data that
            # isn't like the rest of its data and is more like the other histogram data,
            # so the line must be moved
        h1, h2 = self.create_histograms(img, rand_y)

		# 2: Reclassify Points, P(p | H1), P(p | H2) then relabel points
            # Compare 2 histograms and see which color seems like it should belong to
            # the other histogram. Take largest color value of the bigger histogram
            # and see if it matches a color on the smaller histogram. If it does,
            # then the line should be moved until that color is minimized.

        # Calculate which histogram is for the smaller image
        # if (img.shape[1] - random_line_img) > (img.shape[1] / 2.0):
        #     smaller_img_hist = h1
        # else:
        #     smaller_img_hist = h2
        # plt.plot(smaller_img_hist)
        # print(smaller_img_hist)
        # Algorithm:
        # Grab a tuple max_tuple <(x, bin, [R,G,B])> where x is the max number of pixels
        # from the smaller image.
        # past_tuple = None
        # While (smaller_img_h_x_value is greater than zero and is still decreasing):

            # still decreasing means past_x > current_x

            # if line is closer to bottom of image:
                # Move line up a few pxels
            # if line is closer to top of image:
                # Move line down a few pxels
            # Recalculate histograms h1 and h2 and recalculate # of pixels x for
            # smaller image
            # print(new_smaller_img_h_x_value)
            # past_x = new_smaller_img_h_x_value
    def clothing_distribution(self):
        converged = None
        while not converged:
            self.color_em()
		# 3: (Physical Space) K-Means step on theta line, mean of the 2 distributions to
        self.kmeans()

def main():
    # from load_data import LoadData
    # dataset = LoadData('../data_capture/individuals/', (299,299))
    # Testing:
    img = "../../data_capture/individuals/julian.jpeg"
    image = cv2.imread(img)
    image = cv2.resize(image, (299, 299), 0, 0, cv2.INTER_LINEAR)
    image = image.astype(np.float32)
    image = np.multiply(image, 1.0 / 255.0)
    clothing_recognizer = ClothingEM(image)
    clothing_recognizer.color_em(image)


if __name__ == '__main__':
    main()
