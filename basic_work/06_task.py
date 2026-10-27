import cv2
import numpy as np

def morphological_demo(img):
    # 1.先进行预处理：灰度化 与 二值化
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    retval,thresh = cv2.threshold(gray,97,255,cv2.THRESH_BINARY)

    # 2.创建结构元素（Kernel)
    # 5x5 的全1矩阵，代表操作的范围大小
    kernel = np.ones((5,5),np.uint8)

    # 3.腐蚀与膨胀
    eroded = cv2.erode(thresh,kernel,iterations = 1)
    dilated = cv2.dilate(thresh,kernel,iterations = 1)

    cv2.imwrite("output/thresh_robot.jpg",thresh)
    cv2.imwrite("output/erosion_robot.jpg",eroded)
    cv2.imwrite("output/dilation_robot.jpg",dilated)
    print("✅ thresh_robot.jpg 和 erosion_robot.jpg 和 dilation_robot.jpg 已保存至 output_py")
    # 弹窗显示
    # cv2.imshow("thresh_robot.jpg",thresh)
    # cv2.imshow("erosion_robot.jpg",eroded)
    # cv2.imshow("dilation_robot.jpg",dilated)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

if __name__ == "__main__":
    path = "data/robot.jpg"
    img = cv2.imread(path)
    if img is None:
        print("❌ 图片读取失败，请检查路径是否正确")
    else:
        morphological_demo(img)