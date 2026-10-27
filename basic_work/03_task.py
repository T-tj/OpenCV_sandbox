import cv2

def show_original(img):
    cv2.imwrite("output/rainbow_original.jpg",img)
    print("✅ 原图已保存")
    # 弹窗展示
    # cv2.imshow("rainbow.jpg",img)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

def image_conversion(img):
    # 灰度化
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    # 二值化
    _,thresh = cv2.threshold(gray,200,255,cv2.THRESH_BINARY)

    cv2.imwrite("output/gray_rainbow.jpg",gray)
    cv2.imwrite("output/thresh_rainbow.jpg",thresh)
    print("✅ gray_rainbow.jpg 和 thresh_rainbow.jpg 已保存")
    
    # cv2.imshow("gray_rainbow.jpg",gray)
    # cv2.imshow("thresh_rainbow.jpg",thresh)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    

if __name__ == "__main__":
    path = "data/rainbow.jpg"
    img = cv2.imread(path)
    if img is None:
        print("❌ 读取失败，请检查路径是否正确")
    else:
        show_original(img)
        image_conversion(img)