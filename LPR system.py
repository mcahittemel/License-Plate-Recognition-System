import cv2
import numpy as np
import pytesseract
import imutils
import timeit

start = timeit.timeit()

img = cv2.imread("C:\\yedek\\Tez\\Programlar\\tezkodu\\Resimden plaka tespiti\\Tespit\\34 CPD 024.jpeg")                                                    # resimi img adlı değişkene kaydediyoruz
#img = cv2.resize(img,(680,440))                                                    # yeniden boyutlandırma

# Pre-processing

gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)                                         # gri scale görüntü haline çeviriyoruz
filtered = cv2.bilateralFilter(gray,5,75,75)                                        # blur efekt ekliyoruz resimdeki gereksiz kenarları yok etmek için(gürültü azaltma) 5,75,75
histogram_e = cv2.equalizeHist(filtered)                                            # histogram eşitleme ile görüntünün daha iyi bir hale getirilmesi sağlanıyor
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))                          # 1'den oluşan 5 e 5 lik bir matris
morphology = cv2.morphologyEx(histogram_e, cv2.MORPH_OPEN, kernel, iterations=15)   # görüntüye kernel i 15 kere uyguluyoruz
gcikarilmisresim = cv2.subtract(histogram_e,morphology)                             # histogramdan 15 kere kernel uygulanmışsı çıkartıp plaka bölgesinin ön plana çıkmasını sağlıyoruz

# Detection of the License Plate Area

edged = cv2.Canny(gcikarilmisresim,30,250)                                          # Canny edge detetction  30 ve 200 denenmiş iyi sonuçlar alınmış
contours =cv2.findContours(edged,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)             # kenarları bulur
cnts = imutils.grab_contours(contours)                                              # yakalamak, kapmak
cnts = sorted(cnts,key=cv2.contourArea,reverse=True)[:10]                           # bulunan kenarları alanlarına göre sıralar
screen=None                                                                         # kenar sayısını belirtmek için kullanıln bi değşken

for c in cnts:
    epsilon = 0.018*cv2.arcLength(c,True)                                           # konturların yay uzunluğunu bulur 0.018 lik bir hata (yaklaşık)
    approx  = cv2.approxPolyDP(c,epsilon,True)                                      # plaka bölgesindeki dikdörtgeni düzgün birşekile oluturmaya yarıyor
    if len(approx) == 4:                                                            # 4 köşe varsa dikdörtgendir
        screen = approx
        break

mask = np.zeros(gray.shape,np.uint8)                                                # gray formattaki görüntünün aynı boyutlarında siyah bir ekran oluşturur
new_img = cv2.drawContours(mask,[screen],0,(255,255,255),-1)                        # oluşan siyah ekranın plaka bölge kısmını beyaz hale getirir
new_img = cv2.bitwise_and(img,img,mask = mask)                                      # orjinal görüntü ile plaka bölgesinin beyaz olan bölge and ile toplanır

(x,y) = np.where(mask == 255)
(topx,topy) = (np.min(x),np.min(y))
(bottomx,bottomy) = (np.max(x),np.max(y))
cropped = gray[topx:bottomx+1,topy:bottomy+1]                                       # ROI görüntünün sadece plaka bölgesi kırpıldı ben -15 diyerek teseracttan bulunan sondaki  ifadeyi yok etmeye çalıştım

# Character segmentation

ret, binary = cv2.threshold(cropped,110,255,cv2.THRESH_BINARY)                      # otsu gray scale olan bir görüntüyü binarye çeviriyor
binary = cv2.resize(binary,(600,100))                                               # yeniden boyutlandırma
kernel = np.ones((3, 3), np.uint8)                                                  # structur element
binaryerosion = cv2.erode(binary, kernel, iterations=1)                             # beyaz bölgeyi erozyona uğratıyoruz
binarydilation = cv2.dilate(binaryerosion, kernel, iterations=1)                    # opening/closing işlemi uyguluyoruz

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

custom_config = r' -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ-0123456789 --psm 11'
LP = pytesseract.image_to_string(binarydilation, config=custom_config)
print("Detecting License Plate:",LP[:-2])

end = timeit.timeit()
print("Operation time : ",end-start)

cv2.waitKey(0)
cv2.destroyAllWindows()