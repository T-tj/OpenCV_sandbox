import cv2
import numpy as np

def apples_detection(img):
    hsv = cv2.cvtColor(img,cv2.COLOR_BGR2HSV)

    # 因为红色的 HSV 范围跨了0和180两个区间，所以要设两个,设宽松些，包括橙色与淡红
    lower_red1 = np.array([0,100,125])
    upper_red1 = np.array([17,255,255])

    lower_red2 = np.array([160,100,100])
    upper_red2 = np.array([180,255,255])

    # 生成掩膜并合并
    mask1 = cv2.inRange(hsv,lower_red1,upper_red1)
    mask2 = cv2.inRange(hsv,lower_red2,upper_red2)
    mask = cv2.bitwise_or(mask1,mask2)

    # 形态学去噪 (核心)
    # 苹果表面可能有高光反光，导致掩膜上有“黑洞”；叶子缝隙可能漏出红点。
    kernel = np.ones((5,5), np.uint8)
    # 闭运算：填补苹果表面的黑洞
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    # 开运算：消除绿叶上的小红点噪点
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    # 寻找轮廓
    contours,_ = cv2.findContours(mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)

    result_img = img.copy()
    apple_count = 0

    for contour in contours:
        # 5. 面积过滤（去掉图上的 WPS 水印和细碎噪点）
        area = cv2.contourArea(contour)
        if area > 0:  # 过滤掉面积太小的红点（具体数值按图片大小微调）
            
            cv2.drawContours(result_img,contours,-1,(0,255,0),2)
            apple_count += 1

    # 在左上角打印数量
    # 准备文字内容，例如 "Apples: 25"
    text = f"Apple Count: {apple_count}"

    # cv2.putText(图片, 文字, 左下角起点坐标, 字体, 字体大小, 颜色(BGR), 线条粗细)
    cv2.putText(result_img, text, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)
    cv2.imwrite("output/apples_04_detected.jpg",result_img)
    print("✅ output/apple_04_detected.jpg 已保存至 output")

if __name__ == "__main__":
    path = "data/apple_04.jpg"
    img = cv2.imread(path)
    if img is None:
        print("❌ 读取失败，请检查图片路径或文件名")
        exit()
    apples_detection(img)