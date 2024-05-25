import cv2
import numpy as np

# Thiết lập chiều rộng và chiều cao khung hình
width = 540
height = 640

# Mở webcam
cap = cv2.VideoCapture(0)
cap.set(3, width)  # Đặt chiều rộng của khung hình
cap.set(4, height)  # Đặt chiều cao của khung hình
cap.set(10, 100)  # Đặt độ sáng của khung hình

# Hàm để chồng các ảnh lên nhau
def stackImages(scale, imgArray):
    rows = len(imgArray)
    cols = len(imgArray[0])
    rowsAvailable = isinstance(imgArray[0], list)
    width = imgArray[0][0].shape[1]
    height = imgArray[0][0].shape[0]
    if rowsAvailable:
        for x in range(0, rows):
            for y in range(0, cols):
                if imgArray[x][y].shape[:2] == imgArray[0][0].shape[:2]:
                    imgArray[x][y] = cv2.resize(imgArray[x][y], (0, 0), None, scale, scale)
                else:
                    imgArray[x][y] = cv2.resize(imgArray[x][y], (imgArray[0][0].shape[1], imgArray[0][0].shape[0]), None, scale, scale)
                if len(imgArray[x][y].shape) == 2:
                    imgArray[x][y] = cv2.cvtColor(imgArray[x][y], cv2.COLOR_GRAY2BGR)
        imageBlank = np.zeros((height, width, 3), np.uint8)
        hor = [imageBlank] * rows
        hor_con = [imageBlank] * rows
        for x in range(0, rows):
            hor[x] = np.hstack(imgArray[x])
        ver = np.vstack(hor)
    else:
        for x in range(0, rows):
            if imgArray[x].shape[:2] == imgArray[0].shape[:2]:
                imgArray[x] = cv2.resize(imgArray[x], (0, 0), None, scale, scale)
            else:
                imgArray[x] = cv2.resize(imgArray[x], (imgArray[0].shape[1], imgArray[0].shape[0]), None, scale, scale)
            if len(imgArray[x].shape) == 2:
                imgArray[x] = cv2.cvtColor(imgArray[x], cv2.COLOR_GRAY2BGR)
        hor = np.hstack(imgArray)
        ver = hor
    return ver

# Hàm để xử lý trước ảnh
def preProcessing(img):
    imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # Chuyển đổi ảnh sang màu xám
    imgBlur = cv2.GaussianBlur(imgGray, (5, 5), 1)  # Làm mờ ảnh
    imgCanny = cv2.Canny(imgBlur, 100, 100)  # Phát hiện cạnh bằng Canny
    imgDilate = cv2.dilate(imgCanny, np.ones((5, 5)), iterations=2)  # Giãn ảnh
    imgThresh = cv2.erode(imgDilate, np.ones((5, 5)), iterations=1)  # Xói mòn ảnh
    return imgThresh

# Hàm để lấy các đường viền lớn nhất có diện tích lớn nhất
def getContours(img):
    biggest = np.array([])
    maxArea = 0
    contours, hierarchy = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > 5000:
            peri = cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
            if area > maxArea and len(approx) == 4:
                biggest = approx
                maxArea = area
    cv2.drawContours(imgContour, biggest, -1, (255, 0, 0), 20)
    return biggest

# Hàm để sắp xếp lại các điểm
def reorder(myPoints):
    myPoints = myPoints.reshape((4, 2))
    reorderPoints = np.zeros((4, 1, 2), np.int32)
    add = np.sum(myPoints, axis=1)
    reorderPoints[0] = myPoints[np.argmin(add)]
    reorderPoints[3] = myPoints[np.argmax(add)]
    diff = np.diff(myPoints, axis=1)
    reorderPoints[1] = myPoints[np.argmin(diff)]
    reorderPoints[2] = myPoints[np.argmax(diff)]
    return reorderPoints

# Hàm để biến đổi phối cảnh của ảnh
def getWarp(img, biggest):
    biggest = reorder(biggest)
    pts1 = np.float32(biggest)
    pts2 = np.float32([[0, 0], [width, 0], [0, height], [width, height]])
    matrix = cv2.getPerspectiveTransform(pts1, pts2)
    imgOutput = cv2.warpPerspective(img, matrix, (width, height))
    imgCropped = imgOutput[10:imgOutput.shape[0]-10, 10:imgOutput.shape[1]-10]
    imgCropped = cv2.resize(imgCropped, (width, height))
    return imgCropped

while True:
    susscess, img = cap.read()  # Đọc hình ảnh từ webcam
    img = cv2.resize(img, (width, height))  # Thay đổi kích thước ảnh
    imgContour = img.copy()  # Tạo bản sao của ảnh gốc để vẽ đường viền
    imgThresh = preProcessing(img)  # Xử lý trước ảnh
    biggest = getContours(imgThresh)  # Lấy các đường viền lớn nhất

    if biggest.size != 0:
        imgWarped = getWarp(img, biggest)  # Biến đổi phối cảnh của ảnh
        imgArray = ([img, imgThresh],
                    [imgContour, imgWarped])
        cv2.imshow('result', imgWarped)  # Hiển thị ảnh kết quả
    else:
        imgArray = ([img, imgThresh],
                    [img, img])

    stackedImage = stackImages(0.6, imgArray)  # Chồng các ảnh lên nhau
    cv2.imshow("work flow", stackedImage)  # Hiển thị ảnh chồng

    if cv2.waitKey(1) & 0xFF == ord('q'):  # Nhấn phím 'q' để thoát
        break
