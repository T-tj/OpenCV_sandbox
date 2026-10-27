import cv2
import numpy as np

def output_contours_image (img):
    # 预处理
    hsv = cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
    lower_blue = np.array([100,100,70])
    upper_blue = np.array([124,255,255])

    mask_sky = cv2.inRange(hsv,lower_blue,upper_blue)
    # 对掩膜进行取反
    mask_bird = cv2.bitwise_not(mask_sky)
    cv2.imwrite("output/mask_bird.jpg",mask_bird)

    # 轮廓检测
    contours,_ = cv2.findContours(mask_bird,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
    contours_kite_02 = img.copy()
    cv2.drawContours(contours_kite_02,contours,-1,(0,0,255),3)

    cv2.imwrite("output/contours_kite_02.jpg",contours_kite_02)
    print('✅ mask_bird.jpg 和 contours_kite_02.jpg 已保存')
    return contours

def draw_min_rectangle (img,contours):
    img_rect = img.copy()
    if contours:
        # 因为该图有边框，所以要先过滤
        img_area = img.shape[0]*img.shape[1]   # 总面积
        vaild_contours = [c for c in contours if cv2.contourArea(c) < img_area*0.5]  # 面积大于全图90%的直接无视

        if vaild_contours: 
            # 利用max函数寻找面积最大的函数
            largest_contour= max(vaild_contours,key = cv2.contourArea)

            rect = cv2.minAreaRect(largest_contour)
            box = cv2.boxPoints(rect)
            box = np.int32(box)

            cv2.drawContours(img_rect,[box],0,(0,0,255),3)
    cv2.imwrite("output/min-rect_kite_02.jpg",img_rect)
    print("✅ min-rect_kite_02.jpg 已保存")


if __name__ == "__main__":
    path = "data/Black-winged_Kite.jpg"
    img = cv2.imread(path)
    if img is None:
        print("❌ 读取失败，请检查路径是否正确")
    else:
        contours = output_contours_image (img)
        draw_min_rectangle (img,contours)