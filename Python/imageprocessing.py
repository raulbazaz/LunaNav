import numpy as np
import matplotlib.pyplot as plt
import pandas
import cv2

def grayscale(imgpath):
    img = cv2.imread(imgpath)
    cv2.imshow('Original', img)
    gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return gray_image

grayscale(r"Figures\3D.png")
grayscale(r"Figures\2D.png")

