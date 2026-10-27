import cv2
import numpy as np
import os

def detect_apples(img):
    # 1. 转换到 HSV 色彩空间
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    
    # 2. 设定红色的 HSV 范围 (红色跨了 0 和 180 两个区间)
    # 低阈值红色区间
    lower_red1 = np.array([0, 100, 100])
    upper_red1 = np.array([10, 255, 255])
    # 高阈值红色区间
    lower_red2 = np.array([170, 100, 100])
    upper_red2 = np.array([180, 255, 255])
    
    # 3. 生成掩膜并合并
    mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
    mask = cv2.bitwise_or(mask1, mask2)
    
    # 形态学去噪 (核心)
    # 苹果表面可能有高光反光，导致掩膜上有“黑洞”；叶子缝隙可能漏出红点。
    kernel = np.ones((5,5), np.uint8)
    # 闭运算：填补苹果表面的黑洞
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    # 开运算：消除绿叶上的小红点噪点
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    
    # 4. 寻找轮廓
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    result_img = img.copy()
    apple_count = 0
    
    for contour in contours:
        # 5. 面积过滤（去掉图上的 WPS 水印和细碎噪点）
        area = cv2.contourArea(contour)
        if area > 300:  # 过滤掉面积太小的红点（具体数值按图片大小微调）
            
            # 方式一：画矩形框 (boundingRect)
            # x, y, w, h = cv2.boundingRect(contour)
            # cv2.rectangle(result_img, (x, y), (x+w, y+h), (0, 255, 0), 2)
            
            # 方式二：画圆形框 (minEnclosingCircle) -> 苹果是圆的，用圆框
            (x, y), radius = cv2.minEnclosingCircle(contour)

            # 彻底去除高光影响
            # 过滤 1：半径太小不要（干掉高光小圆圈）
            if radius < 15:
                continue
            # 过滤 2：不够圆不要（干掉不规则的反光碎块）
            circle_area = np.pi * (radius ** 2)
            circularity = area / circle_area 
            if circularity < 0.5:  
                continue

            center = (int(x), int(y))
            radius = int(radius)
            cv2.circle(result_img, center, radius, (0, 255, 0), 2)
            
            apple_count += 1
    
    # 在左上角打印数量
    # 准备文字内容，例如 "Apples: 25"
    text = f"Apple Count: {apple_count}"

    # cv2.putText(图片, 文字, 左下角起点坐标, 字体, 字体大小, 颜色(BGR), 线条粗细)
    cv2.putText(result_img, text, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)

    return result_img, apple_count

if __name__ == "__main__":
    path = "data/apple_01.jpg"
    img = cv2.imread(path)
    if img is None:
        print("❌ 读取失败，请检查图片路径或文件名")
        exit()

    result_img, count = detect_apples(img)
        
    cv2.imwrite("output/apple_0_detected.jpg", result_img)
    print(f"✅ 识别完成！共找到 {count} 个苹果。")
    print("✅ 已保存 apple_01_detected.jpg 至 output")