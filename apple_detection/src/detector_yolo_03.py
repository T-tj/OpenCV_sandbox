from ultralytics import YOLO
import cv2

def detect_apples_yolo(img_path,out_path,model_path):
    model = YOLO(model_path)
    print(model.names)
    img = cv2.imread(img_path) 
    if img is None:
        print("❌ 读取失败，请检查路径是否正确")
        exit()

    results = model(img,conf = 0.15,iou = 0.35,imgsz = 2560)
    result = results[0]

    apple_count = 0
    for box in result.boxes:
        # 获取左上角和右下角的坐标
        x1,y1,x2,y2 = map(int,box.xyxy[0])
        # 获取置信度
        conf = float(box.conf[0])
        # 获取ID
        cls_id = int(box.cls[0])
        cls_name = model.names[cls_id]
        # 绘制矩形框
        cv2.rectangle(img,(x1,y1),(x2,y2),(0,255,0),2)

        # 在框的左上角标出数据
        label = f"{cls_name} {conf:2f}"
        cv2.putText(img,label,(x1,y1-10),cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,255,0),2)

        # 泛化标准，橙色与淡黄色苹果用 orange 代替
        if cls_name in ['apple','orange']:  
            apple_count += 1

    text = f"Apple Count: {apple_count}"
    cv2.putText(img,text,(20,50),cv2.FONT_HERSHEY_SIMPLEX,1.5,(0,0,255),3)
    cv2.imwrite(out_path,img)
    print(f"✅ YOLO 识别完成！共找到 {apple_count} 个苹果，结果已保存至 {out_path}")


if __name__ == "__main__":
    img_path = 'data/apple_04.jpg'
    out_path = 'output/yolo_apple_04.jpg'
    model_path = 'yolov8n.pt'
    detect_apples_yolo(img_path,out_path,model_path)