'''
Created on March 12th, 2018
author: Julian Weisbord
sources:
description:
'''
import sys
from clothing.clothing_em import ClothingEM
from body_type.body_type_classifier import BodyTypeClassifier
from load_data import LoadData

DEFAULT_LIVE_IMG_PATH = '../data_capture/live_input/imgs/'
DEFAULT_INDIVIDUAL_IMG_PATH = '../data_capture/individuals/'
PREDICTION_THRESHOLD = 70
CLOTHING_WEIGHT = .5
BODY_TYPE_WEIGHT = .5
IMAGE_WIDTH = 299
IMAGE_HEIGHT = 299

def individual_hist():
    pass

def individual_body_type():
    pass

class Predict():
    def __init__(self, img_path):
        self.img_path = img_path

    def compare_individual_w_clothing(self, unkown_clothing_distr, individual_clothing_distr):
        pass

    def compare_individual_w_body(self, unkown_body_distr, individual_body_distr):
        pass


def main():

    if len(sys.argv == 2):
        img_path = sys.argv[1]
    else:
        img_path = DEFAULT_LIVE_IMG_PATH

    prediction = Predict(img_path)

    p_clothing = []
    p_body_type = []
    probability = None
    # If there was an update to the live img data location
        # Get live image/vid data
    live_imgs = LoadData(DEFAULT_LIVE_IMG_PATH, (IMAGE_WIDTH, IMAGE_HEIGHT))
    live_frame_dataset = live_imgs.get_last_n_frames(n=1)
    # Get the initial photos of individuals
    initial_people = LoadData(DEFAULT_INDIVIDUAL_IMG_PATH)
    person_dataset = initial_people.get_all_frames()

    # Given an individual's clothing hist and body type:
    # Compare that to the unkown body type and clothing histogram
    clothing_prob = ClothingEM()
    unkown_clothing_prob = ClothingEM(live_frame_dataset)
    unkown_clothing_dist = unkown_clothing_prob.clothing_distribution()
    body_prob = BodyTypeClassifier()


    for person in person_dataset:
        clothing_prob.add_frames(person)
        individual_clothing_dist = clothing_prob.clothing_distribution()
        # Compare each individual distribution with the unknown distribution and add it to arrays
        clothing_prediction = prediction.compare_individual_w_clothing(unkown_clothing_dist, individual_clothing_dist)

        body_prediction = prediction.compare_individual_w_body(unkown_body_distr, individual_body_distr)
        p_clothing.append({person:clothing_prediction})
        p_body_type.append({person:body_prediction})


        # output the probablility that it is them.
            # P(Individual | Data) =  a * P(Individual | Clothing) + b *P(Individual | Body Type)
        # If it is above a certain threshold, add the person to a cache

if __name__ == '__main__':
    main()
