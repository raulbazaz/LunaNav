import cv2

def edgeDetection(imgname, imgpath):
    imgload = cv2.imread(imgpath, cv2.IMREAD_GRAYSCALE)
    blurred = cv2.GaussianBlur(imgload,(5,5), 0)
    edgeDetection = cv2.Canny(blurred,50,150)
    cv2.imwrite(f"{imgname}_edgeDet.png", edgeDetection)
    return edgeDetection

edgeDetection("2D", '2D_Clahe.png')
edgeDetection("3D", '3D_Clahe.png')

