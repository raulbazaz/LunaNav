import clahe
import cv2


def claheprocessing(imgpath):
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    enhanced = clahe.apply(imgpath)
    cv2.imwrite(imgpath, enhanced)
    return enhanced


