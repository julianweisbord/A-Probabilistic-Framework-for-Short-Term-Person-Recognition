'''
Created on March 8th, 2018
author: Julian Weisbord
sources: https://docs.opencv.org/3.1.0/d1/db7/tutorial_py_histogram_begins.html
description: Clothing prediction portion of the model, calculates the likelihood
                that the live data contains someone from the database of people
                by breaking the feed images into color histograms.
'''

import sys
import random
import glob
import numpy as np
from sys import maxint
import cv2
from matplotlib import pyplot as plt

IMG_WIDTH = 299
IMG_HEIGHT = 299
VERBOSITY = 2

class ClothingEM():
    '''
    description: Uses E&M to cluster the clothing of people seen by the robot.
    input: frame_dataset <Dataset object> the input dataset,
               video <boolean> whether or not the input is video or photo data.
    '''

    def __init__(self, frame_dataset=None, video=False):
        self.images = frame_dataset[0]
        self.image_names = frame_dataset[1]
        self.person_locations = frame_dataset[2]
        if video:
            self.video = video
            # Convert video to a series of frames

    def add_frames(self, dataset):
        '''
        description: Add a dataset of camera frames.
        input: dataset <Dataset object>
        '''
        frames = []
        names = []
        locations = []
        for person_data in dataset:
            self.images.append(dataset[0])
            self.image_names.append(dataset[1])
            self.person_locations.append(dataset[0])

    def crop_to_person(self):
        '''
        description: Crop a given data frame to isolate just the person in it.
        return: x1,x2,y1,y2 <int> the four corners of the cropped image.
        '''
        # Simulate this function for now with constant values
        # TODO: Implement this with grabcut Algorithm?
        x1 = 100
        x2 = 192
        y1 = 16
        y2 = 272
        return x1, x2, y1, y2

    def create_histograms(self, img, line_height, x1, x2, y1, y2):
        '''
        description: Create a color histogram above and below the random line (line_height).
        input: img <numpy.ndarray>, line_height <int> height of randomly generated line,
                       x1, x2, y1, y2 <int> 4 corners of input image.
        return:
        '''

        if VERBOSITY >=2:
            l = cv2.imshow('Random Line Splitting Clothing', img)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
            print("img shape: ", img.shape)

        # Split image into upper and lower masks
        mask_upper_cropped = np.zeros(img.shape[:2], np.uint8)
        mask_upper_cropped[y1:line_height, x1:x2] = 255
        upper_masked_img_cropped = cv2.bitwise_and(img, img, mask=mask_upper_cropped)

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
            upper_plot.imshow(cv2.cvtColor(upper_masked_img_cropped, cv2.COLOR_RGB2BGR))
            lower_plot.imshow(cv2.cvtColor(lower_masked_img_cropped, cv2.COLOR_RGB2BGR))
        # Create and polot histograms for r,g,b collor values.
        colors = ("b", "g", "r")
        features_h1 = []
        features_h2 = []
        h1_colors = []
        h2_colors = []
        for i,col in enumerate(colors):
            h1 = cv2.calcHist([img], [i], mask_upper_cropped, [256], [0, 1])
            h1_colors.append([arr[0] for arr in h1])
            h2 = cv2.calcHist([img], [i], mask_lower_cropped, [256], [0, 1])
            h2_colors.append([arr[0] for arr in h2])
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
        h1_rgb = []
        h2_rgb = []
        for i in range(len(h1)):
            h1_rgb.append([single_color_vector[i] for single_color_vector in h1_colors])
            h2_rgb.append([single_color_vector[i] for single_color_vector in h2_colors])
        print("h1_rgb: ", h1_rgb)
        print("len h1_rgb: ", len(h1_rgb))
        h1_rgb = np.array(h1_rgb)
        h2_rgb = np.array(h2_rgb)
        return h1_rgb, h2_rgb


    def distance(self, point, center, axis=1):
        pass

    def label_data(self, data, C):
        '''
        description: For each row in the data, calculate the distance between that
                        row and each centroid.
        input: data <numpy array>, C <list of k numpy arrays>
        return: labels <list> of each centroid number corresponding to
                each row in the data.
        '''
        labels = []
        for n, row in enumerate(data):
            best_dist = maxint
            best_c_k = 0
            for c_k, centroid in enumerate(C):
                row_dist = distance(row, centroid)
                if row_dist < best_dist:
                    best_dist = row_dist
                    best_c_k = c_k
            labels.append(best_c_k)
        return labels

    def expectation(self, h1_rgb, h2_rgb):
        '''
        description: Assign points to clusters and calculate mean.
        input: h1_rgb <numpy array> top color histogram, h2_rgb <numpy array> bottom color histogram.
        return mean rgb clusters
        '''
        # Calculate mu (mean r,g,b centroid) and sigma (covariance matrix) for each cluster
        mean_h1 = np.mean(h1_rgb, axis = 0)
        mean_h2 = np.mean(h2_rgb, axis = 0)
        print("mean 1", mean_h1)
        print("mean 2", mean_h2)
        covariance_h1 = np.cov(h1_rgb)
        covariance_h2 = np.cov(h2_rgb)
        print("covariance h1", covariance_h1)
        print("covariance h2", covariance_h2)
        print("len covariance h1", len(covariance_h1))
        return [(mean_h1, covariance_h1), (mean_h2, covariance_h2)]

    def color_em(self, img):
        '''
        description: Performs Expecation Maximization with 2 clusters
                        on the input image.
        input: img <numpy array> the image fed to this classifier.
        '''
        # Crop image so that only the person is in the image:
        x1, x2, y1, y2 = self.crop_to_person()
        rand_y = random.randint(y1, y2 + 1)

        random_line_img = cv2.line(img, (0, rand_y), (IMG_WIDTH, rand_y), (0, 0, 250), 4)

        # Save random line image
        random_line_img_out = np.multiply(random_line_img, 255.0)
        random_line_img_out = random_line_img_out.astype('uint8')
        cv2.imwrite('../../data_capture/manipulated_images/line.jpeg', random_line_img_out)

        cropped_img = img[y1:y2, x1:x2]
        cropped_img = np.multiply(cropped_img, 255.0)
        cropped_img = cropped_img.astype('uint8')
        cv2.imwrite('../../data_capture/manipulated_images/cropped_line.jpg', cropped_img)

        # 1: Build color histograms, Histogram1 (H1) is above the theta line,
            # and Histogram 2 (H2) is below. One histogram will likely contain data that
            # isn't like the rest of its data and is more like the other histogram data,
            # so the line must be moved
        line_pos = rand_y
        prior_line_pos = 0
        # Need to loop

        h1_rgb, h2_rgb = self.create_histograms(img, rand_y, x1, x2, y1, y2)
        self.expectation(h1_rgb, h2_rgb)


    def clothing_distribution(self):
        converged = None
        while not converged:
            self.color_em()

def main():
    img = "../../data_capture/individuals/julian.jpg"
    image = cv2.imread(img)
    image = cv2.resize(image, (299, 299), 0, 0, cv2.INTER_LINEAR)
    image = image.astype(np.float32)
    image = np.multiply(image, 1.0 / 255.0)
    clothing_recognizer = ClothingEM(image)
    clothing_recognizer.color_em(image)


if __name__ == '__main__':
    main()
