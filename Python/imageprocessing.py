import numpy as np
import matplotlib.pyplot as plt
import pandas
import cv2

def grayscale(imgname, imgpath):
    img = cv2.imread(imgpath)
    gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    cv2.imwrite(f"{imgname}_grayscaled.png", gray_image)
    return gray_image

grayscale('2D', 'Figures/2D.png')
grayscale('3D', 'Figures/3D.png')



