import cv2

image = cv2.imread(r'D:\Admin\Documents\Year_4\AMLS\Assessment\dataset_AMLS_20-21\cartoon_set\img\0.png')
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
cv2.imshow('Original image', image)
cv2.imshow('Gray image', gray)
cv2.waitKey(0)
cv2.destroyAllWindows()
