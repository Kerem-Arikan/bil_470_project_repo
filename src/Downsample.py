import numpy as np
import tkinter as tk
from tkinter import filedialog
import matplotlib.pyplot as plt
import cv2 

class Downsample:

    def __init__(self, feature_map):
        self.feature_map = feature_map


    def downsample(self,row_size,column_size):
        
        res = cv2.resize(self.feature_map, dsize=(row_size,column_size),interpolation = cv2.INTER_CUBIC)
        
        print("Downsized image row = ",row_size," - column =",column_size)

        return res
