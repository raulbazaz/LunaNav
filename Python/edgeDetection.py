import cv2

def edgeDetection(imgname, imgpath):
    clahe = cv2.imread(imgpath, cv2.IMREAD_GRAYSCALE)
    edgeDetection = cv2.Canny(clahe,50,150)
    cv2.imwrite(f"{imgname}_edgeDet.png", edgeDetection)
    return edgeDetection

edgeDetection("2D", '2D_Clahe.png')
edgeDetection("3D", '3D_Clahe.png')

