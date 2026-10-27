import cv2
import numpy as np

def output_contours_image (img):
    # 预处理：灰化，高斯，边缘
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray,(5,5),0)
    edge = cv2.Canny(blurred,50,150)

    # 膨胀让断线相连
    kernel = np.ones((5,5),np.uint8)
    dilated_edge = cv2.dilate(edge,kernel,iterations = 1)

    # 轮廓检测
    contours,_ = cv2.findContours(dilated_edge,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
    contours_img = img.copy()
    cv2.drawContours(contours_img,contours,-1,(0,0,255),3)

    cv2.imwrite("output/contours_cat.jpg",contours_img)
    print("✅ contours_cat.jpg 已保存")
    # 弹窗展示
    # cv2.imshow("contours_cat.jpg",contours_img)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    return contours

def draw_min_rectangle (img,contours):
    # 将检测到的轮廓转换为最小外接矩形
    img_rect = img.copy()
    # 过滤掉面积太小的轮廓
    for contour in contours:
        if cv2.contourArea(contour) > 50:
            rect = cv2.minAreaRect(contour)
            box = cv2.boxPoints(rect)
            box = np.int32(box)
            cv2.drawContours(img_rect,[box],0,(0,0,255),3)
    cv2.imwrite("output/min-rect_cat.jpg",img_rect)
    print("✅ min-rect_cat.jpg 已保存")
    # cv2.imshow("min-rect_cat.jpg",img_rect)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

if __name__ == "__main__":
    path = "data/cat.jpg"
    img = cv2.imread(path)
    if img is None:
        print("❌ 读取失败，请检查路径是否正确")
    else:
        contours = output_contours_image (img)
        draw_min_rectangle (img,contours)