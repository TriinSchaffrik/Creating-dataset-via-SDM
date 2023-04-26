import os
import random

import cv2
import json
import numpy as np


def rgb2indexes(img, indexes):
    pi = ""
    categories = dict()
    new_pix_value = 0
    # look through every pixel on image
    for row in range(len(img)):
        for pix in range(len(img[row])):
            # change old indexes to new indexes
            if str(img[row][pix][0]) != pi:
                pi = str(img[row][pix][0])
                # already seen that category before
                if pi in categories.keys():
                    new_pix_value = categories[pi]
                # we have value for this category
                elif pi in indexes.keys():
                    values = indexes[pi]
                    new_pix_value = random.choice(values)
                    categories[pi] = new_pix_value
                # we do not have value for that category
                else:
                    new_pix_value = 0
            img[row][pix] = np.ones(3, dtype=int) * new_pix_value
    return img


def json2dict(filename):
    with open(filename) as f:
        d = json.load(f)
    return d


def main():
    indexes = json2dict('categories_coco2ade20k.json')  # keys: old indexes (str) ; values: new indexes (int)
    input_dir = "./annotations_2"  # masked images directory
    output_dir = "./annotations"  # dir where new img are saved
    args = parse_args()
    all_images = [file for file in os.listdir(input_dir) if file.endswith('.png')]
    for image in all_images:
        img = cv2.imread(input_dir + "/" + image)  # read
        print(image)
        indexed_img = rgb2indexes(img, indexes)  # convert
        cv2.imwrite(output_dir + "/" + image, indexed_img)  # save
    print("All done.")


if __name__ == "__main__":
    main()
