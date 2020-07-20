import pandas as pd
import os
import cv2 
class Data(object):

    def __init__(self, csv_file, path, size, target=[1,0]):

        samples = pd.read_csv(csv_file)
        samples = samples[samples.target.isin(target)].head(size)

        self.csv = samples
        self.image_names = samples.image_name.array
        self.index = 0
        self.length = len(self.image_names)
        self.path = path
        self.target = target
        self.csv_file = csv_file
        self.size = size

    def nextSample(self):
        name = self.image_names[self.index]
        self.index += 1
        filepath = self.path + name + ".jpg"
        feature_map = cv2.imread(filepath)
        feature_map = cv2.cvtColor(feature_map, cv2.COLOR_BGR2GRAY)
        feature_map = cv2.resize(feature_map, dsize=(514,514), interpolation=cv2.INTER_CUBIC)
        return feature_map


    def hasSample(self):
        return (self.index < self.length)

