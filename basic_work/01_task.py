import cv2

def show_original(img):
    # 弹窗展示
    # cv2.imshow("Black_gun",img)
    # cv2.waitKey(0)
    # cv2.destoryALLWindows()
    cv2.imwrite("output/black_original.jpg",img)
    print("✅ 原图已保存")

def resize_long_edge (img,scale = 0.5):
    h,w = img.shape[:2]
    new_h = int(h*scale)
    resized = cv2.resize(img,(w,new_h))
    
    # cv2.imshow("Resize_black.jpg",resized)
    # cv2.waitKey(0)
    # cv2.destoryALLWindows()
    cv2.imwrite("output/resize_black_gun.jpg",resized)
    print("✅ 缩放图已保存，原尺寸：{w}x{h} -> 新尺寸：{w}x{new_h}")

def rotate_clockwise_90(img):
    rotated = cv2.rotate(img,cv2.ROTATE_90_CLOCKWISE)

    # cv2.imshow("Rotate_black.jpg",rotated)
    # cv2.waitKey(0)
    # cv2.destoryALLWindows()
    cv2.imwrite("output/rotale_black.jpg",rotated)
    print("✅ 90度旋转图已保存")


if __name__ == "__main__":
    path = 'data/black_gun.jpg'
    img = cv2.imread(path)
    if img is None:
        print("❌ 读取失败，请检查路径")
    else:
        show_original(img)
        resize_long_edge (img,scale = 0.5)
        rotate_clockwise_90(img)