import cv2

for i in xrange(5):
	m1 = cv2.imread("img_"+str(i)+".png")
	m2 = cv2.resize(m1, None, fx=0.5, fy=0.5)
	cv2.imwrite("img_"+str(i)+".png", m2)
