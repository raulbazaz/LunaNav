import cv2
import numpy as np

def edgedetection(imgname, imgpath):
    imgload = cv2.imread(imgpath, cv2.IMREAD_GRAYSCALE)
    blurred = cv2.GaussianBlur(imgload,(5,5), 0)
    edgeDetection = cv2.Canny(blurred,50,150)
    cv2.imwrite(f"{imgname}_edgeDet.png", edgeDetection)
    return edgeDetection

def thresholding(imgname, imgpath):
    imgload = cv2.imread(imgpath, cv2.IMREAD_GRAYSCALE)
    thresh = cv2.adaptiveThreshold(imgload, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY, 11, 2)
    cv2.imwrite(f"{imgname}_thresh.png", thresh)
    return thresh

def cleanNoise(imgname, imgpath):
    imgload = cv2.imread(imgpath, cv2.IMREAD_GRAYSCALE)
    kernel = np.ones((3, 3), np.uint8)
    cleaned = cv2.morphologyEx(imgload, cv2.MORPH_CLOSE, kernel, iterations=2)
    cleaned_final = cv2.morphologyEx(cleaned, cv2.MORPH_OPEN, kernel, iterations=1)
    cv2.imwrite(f"{imgname}_cleaned_final.png", cleaned_final)
    cv2.imwrite(f"{imgname}_cleaned.png", cleaned)
    return cleaned

cleanNoise('2D', '2D_thresh.png')
cleanNoise('3D', '3D_thresh.png')




