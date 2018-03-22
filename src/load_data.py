'''
Created on March 12th, 2018
author: Julian Weisbord
sources:
description:
'''

import os
import glob
import numpy as np
import cv2
import data_capture.person_localization as locate


class LoadData():
    '''
    load and prepare live img/video data or individual image data
    '''

    def __init__(self, frame_path, image_size):
        self.person_location = []
        self.frame_path = frame_path
        self.image_size = image_size
        self.frames, self.image_names = self.set_all_frames()


    def get_last_file_added(self):
        return last_added

    def get_last_n_frames(self, n=1):
        '''

        '''
        return images, image_names, location

    def get_all_frames(self):
        return self.frames

    def add_files(self):
        '''
        Add images into the dataset that have been recently
            loaded into the live image directory. Threading
        '''
        # # Locate where the person is in the image
        # self.person_location.append(locate(image))
        pass

    def set_all_frames(self):

        img_names = []  # img file base path, the png file
        images = []
        files = glob.glob(self.frame_path + '*')
        for img in files:
            image = cv2.imread(img)
            image = cv2.resize(image, (self.image_size[0], self.image_size[1]), 0, 0, cv2.INTER_LINEAR)
            image = image.astype(np.float32)
            image = np.multiply(image, 1.0 / 255.0)
            images.append(image)

            # Locate where the person is in the image
            self.person_location.append(locate(image))

            flbase = os.path.basename(img)  # base path of image
            img_names.append(flbase)
        images = np.array(images)
        img_names = np.array(img_names)
        print("Image Name: {}".format(img_names))
        return images, img_names, location

    def delete_frames(self):
        ''' Delete old files in the folder '''
        pass
