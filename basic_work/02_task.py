import cv2

def draw_patrick_shapes(img):
    process_img = img.copy()

    #图像绘制
    cv2.circle(process_img,(350,60),37,(0,0,255),-1)
    cv2.line(process_img,(70,300),(570,340),(0,255,0),3)
    cv2.rectangle(process_img,(160,340),(390,490),(255,0,0),-1)
    cv2.circle(process_img,(265,260),40,(0,255,255),3)
    cv2.rectangle(process_img,(305,305),(360,225),(255,0,255),3)

    cv2.imwrite("output/drawn_patrick.jpg",process_img)
    
    # cv2.imshow("drawn_patrick.jpg",process_img)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    

if __name__ == "__main__":
    path = 'data/patrick.jpg'
    img = cv2.imread(path)
    if img is None:
        print("❌ 读取失败，请检查路径是否正确")
    else:
        draw_patrick_shapes(img)
        print(img.shape)
        print("✅ 操作成功，绘制结果已保存")
       