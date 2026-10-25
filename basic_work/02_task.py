import cv2

def draw_patrick_shapes(img):
    process_img = img.copy()

    cv2.circle(process_img,(400,60),8,(0,0,255),-1)
    cv2.line(process_img,(120,400),(520,420),(0,255,0),3)
    cv2.rectangle(process_img,(250,280),(430,400),(255,0,0),-1)
    cv2.circle(process_img,(310,200),40,(0,255,255),2)
    cv2.rectangle(process_img,(400,160),(470,240),(255,0,255),2)

    # cv2.imshow("Resize_black.jpg",resized)
    # cv2.waitKey(0)
    # cv2.destoryALLWindows()
    cv2.imwrite("output/drawn_patrick.jpg",process_img)
    

if __name__ == "__main__":
    path = 'data/patrick.jpg'
    img = cv2.imread(path)
    if img is None:
        print("❌ 读取失败，请检查路径是否正确")
    else:
        draw_patrick_shapes(img)
        print("✅ 操作成功，绘制结果已保存")