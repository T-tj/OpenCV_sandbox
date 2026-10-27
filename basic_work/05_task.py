import cv2

def image_conversion(img):
    # 预处理：灰化，高斯，边缘
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray,(5,5),0)

    edges = cv2.Canny(blurred,50,150)
    contours,_ = cv2.findContours(edges,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)

    # 轮廓检测
    img_contours = img.copy()
    cv2.drawContours(img_contours,contours,-1,(0,0,255),1)

    cv2.imwrite("output/contours_house.jpg",img_contours)
    cv2.imwrite("output/gaussian_house.jpg",blurred)
    print("✅ contours_house.jpg 和 gaussian_house.jpg 已保存")
    
    # 弹窗显示
    # cv2.imshow("contours_house.jpg",img_contours)
    # cv2.imshow("gaussian_house.jpg",blurred)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

if __name__ == "__main__":
    path = "data/house.jpg"
    img = cv2.imread(path)
    if img is None:
        print("❌ 读取失败，请检查路径是否正确")
    else:
        image_conversion(img)