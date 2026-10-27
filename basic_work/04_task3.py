import cv2
import numpy as np
import os

def get_bird_box(img):
    # 1. 转 HSV
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    lower_blue = np.array([90, 50, 50])
    upper_blue = np.array([130, 255, 255])
    mask_sky = cv2.inRange(hsv, lower_blue, upper_blue)
    
    # 2. 取反，得到鸟（白）和背景（黑）
    mask_bird = cv2.bitwise_not(mask_sky)
    
    # 核心：斩断外边框
    # 强行把图片四周 10 个像素涂黑，彻底消灭那个烦人的外围白框
    mask_bird[0:10, :] = 0
    mask_bird[-10:, :] = 0
    mask_bird[:, 0:10] = 0
    mask_bird[:, -10:] = 0

    # 顺便把中间处理的黑白图也存下来，确保没白边了
    cv2.imwrite("output/not-white-edge_mask.jpg",mask_bird)
    print("✅ not-white-edge_mask.jpg 已保存")
    
    # 3. 形态学开运算：切断电线，只留大块头的鸟
    # 核越大，吃掉的小细节越多。15x15 足够把细细的电线腐蚀掉，同时保留鸟的身体。
    kernel = np.ones((15, 15), np.uint8)
    mask_bird = cv2.morphologyEx(mask_bird, cv2.MORPH_OPEN, kernel)
    # 4. 寻找轮廓（用 RETR_EXTERNAL 只找最外层，避免内部羽毛干扰）
    contours, _ = cv2.findContours(mask_bird, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    img_rect = img.copy()
    
    if contours:
        # 5. 因为白框没了，现在面积最大的绝对是鸟（和连着它的电线）
        largest_contour = max(contours, key=cv2.contourArea)
        
        # 6. 获取倾斜的最小外接矩形
        rect = cv2.minAreaRect(largest_contour)
        box = cv2.boxPoints(rect)
        box = np.int32(box)
        
        # 画框
        cv2.drawContours(img_rect, [box], 0, (0, 0, 255), 3)
            
    return img_rect

if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.abspath(__file__))
    path = os.path.join(base_dir, '../data/Black-winged_Kite.jpg')
    img = cv2.imread(path)
    
    if img is not None:
        result_img = get_bird_box(img)
        out_dir = os.path.join(base_dir, '../output')
        
        cv2.imwrite(os.path.join(out_dir, 'bird_perfect_rect.jpg'), result_img)
        print("✅ 斜框已生成：bird_perfect_rect.jpg")