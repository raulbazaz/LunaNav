import cv2

def claheprocessing(imgname, imgpath):
    img = cv2.imread(imgpath, cv2.IMREAD_GRAYSCALE)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    enhanced = clahe.apply(img)
    cv2.imwrite(f'{imgname}_Clahe.png', enhanced)
    return enhanced

claheprocessing("2D", '2D_grayscaled.png')
claheprocessing("3D",'3D_grayscaled.png')





