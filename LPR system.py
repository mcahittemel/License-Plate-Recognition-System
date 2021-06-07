import cv2
import numpy as np
import pytesseract
import imutils
import timeit

start = timeit.timeit()

img = cv2.imread("C:\\yedek\\Tez\\Programlar\\tezkodu\\GitHub\\Image\\34 LZ 6622.jpg")

# Pre-processing

gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)                                         # converts to gray scale image
filtered = cv2.bilateralFilter(gray,5,75,75)                                        # Adds a blur effect to remove unnecessary edges in the image (noise reduction)
histogram_e = cv2.equalizeHist(filtered)                                            # Improved image with histogram equalization
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))                          # A 5 by 5 matrix of 1
morphology = cv2.morphologyEx(histogram_e, cv2.MORPH_OPEN, kernel, iterations=15)   # applying the kernel to the image 15 times
gcikarilmisresim = cv2.subtract(histogram_e,morphology)                             # 15 times kernel applied image is removed from the histogram and the plate region is brought to the fore

# Detection of the License Plate Area

edged = cv2.Canny(gcikarilmisresim,30,250)                                          # Edges are detected with Canny edge detection.
contours =cv2.findContours(edged,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)             # finds edges
cnts = imutils.grab_contours(contours)                                              # catch, grab the countours
cnts = sorted(cnts,key=cv2.contourArea,reverse=True)[:10]                           # The found edges are sorted by their area.
screen=None                                                                         # a variable used to specify the number of sides

for c in cnts:
    epsilon = 0.018*cv2.arcLength(c,True)                                           # finds the arc length of the contours with an error of 0.018 (approximately)
    approx  = cv2.approxPolyDP(c,epsilon,True)                                      # It serves to form the rectangle in the plate region properly.
    if len(approx) == 4:                                                            # It is a rectangle if it has 4 corners
        screen = approx
        break

mask = np.zeros(gray.shape,np.uint8)                                                # creates a black screen with the same dimensions of the gray format image
new_img = cv2.drawContours(mask,[screen],0,(255,255,255),-1)                        # makes the plate region part of the resulting black screen white
new_img = cv2.bitwise_and(img,img,mask = mask)                                      # The white region of the plate region with the original image is summed with and

(x,y) = np.where(mask == 255)
(topx,topy) = (np.min(x),np.min(y))
(bottomx,bottomy) = (np.max(x),np.max(y))
cropped = gray[topx:bottomx+1,topy:bottomy+1]                                       # Only the plate region of the ROI image was cropped, I tried to remove the last expression found in the teseract by saying -15

# Character segmentation

ret, binary = cv2.threshold(cropped,110,255,cv2.THRESH_BINARY)                      # OTSU converts a grayscale image to binary
binary = cv2.resize(binary,(600,100))                                               # resize
kernel = np.ones((3, 3), np.uint8)                                                  # structur element
binaryerosion = cv2.erode(binary, kernel, iterations=1)                             # the white zone is eroded
binarydilation = cv2.dilate(binaryerosion, kernel, iterations=1)                    # the white zone is dilated

cv2.imshow("Original License Plate",img)
#cv2.imwrite("Original_image.jpg",img)
cv2.imshow("Gray License Plate",gray)
#cv2.imwrite("Gray-scale.jpg",gray)
cv2.imshow("Filtered License Plate",filtered)
#cv2.imwrite("Filtered.jpg",filtered)
cv2.imshow("histogram eşikleme",histogram_e)
#cv2.imwrite("Histogram_Esikleme.jpg",histogram_e)
cv2.imshow("Morphology",morphology)
#cv2.imwrite("Morphology.jpg",morphology)
cv2.imshow("goruntuden cikarilmis resim",gcikarilmisresim)
#cv2.imwrite("Cikarilmisresim.jpg",gcikarilmisresim)
cv2.imshow("Egded License Plate",edged)
#cv2.imwrite("Edges.jpg",edged)
cv2.imshow("Mask License Plate",mask)
#cv2.imwrite("mask.jpg",mask)
cv2.imshow("New License Plate",new_img)
#cv2.imwrite("New_image.jpg",new_img)
cv2.imshow("Cropped License Plate",cropped)
#cv2.imwrite("Cropped.jpg",cropped)
cv2.imshow("Binary",binary)
#cv2.imwrite("Binary.jpg",binary)
cv2.imshow("binaryerosion",binaryerosion)
#cv2.imwrite("Binaryerosion.jpg",binaryerosion)
cv2.imshow("binarydilation",binarydilation)
#cv2.imwrite("Binarydilation.jpg",binarydilation)

# Caharacter Recognition with Tesseract-OCR

custom_config = r' -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ-0123456789 --psm 11' # consist of a white list for chracter recognition
LP = pytesseract.image_to_string(binarydilation, config=custom_config)
print("Detecting License Plate:",LP[:-2])

end = timeit.timeit()
print("Operation time : ",end-start)

cv2.waitKey(0)
cv2.destroyAllWindows()