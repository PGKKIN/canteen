import cv2
import time
import torch
from ultralytics import YOLO

# โหลดโมเดล
model_path = r'D:\my_work\canteen\runs\detect\human_final\weights\best.pt'
model = YOLO(model_path)
model.to('cuda')

video_path = r'D:\my_work\canteen\video.mp4' 
cap = cv2.VideoCapture(video_path)

# ตั้งค่าการข้ามเฟรม
frame_count = 0
SKIP_FRAMES = 2  # ข้าม 2 เฟรม ทำงาน 1 เฟรม (ปรับเลขนี้ได้ 1-3)

prev_frame_time = 0

print(f"🚀 เริ่มรันแบบ Frame Skipping (ข้ามทีละ {SKIP_FRAMES} เฟรม)...")

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break

    frame_count += 1
    
    # 🔥 ถ้าไม่ใช่คิวทำงาน ให้ข้ามไปเลย (ช่วยลดภาระ CPU)
    if frame_count % (SKIP_FRAMES + 1) != 0:
        continue

    # ลดขนาดภาพ
    frame = cv2.resize(frame, (1280, 720))

    # ส่งเข้า GPU
    results = model.track(frame, persist=True, conf=0.3, iou=0.5, verbose=False, device=0)

    # วาดผลลัพธ์
    if results[0].boxes.id is not None:
        boxes = results[0].boxes.xyxy.cpu().numpy().astype(int)
        ids = results[0].boxes.id.cpu().numpy().astype(int)
        
        for box, id in zip(boxes, ids):
            # วาดแค่เส้นบางๆ พอ เพื่อความเร็ว
            cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 2)
            # ไม่ต้องวาด ID ทุกคนก็ได้ถ้ารก (หรือวาดให้เล็กลง)
            # cv2.putText(frame, f"{id}", (box[0], box[1]-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)

    # คำนวณ FPS
    new_frame_time = time.time()
    if prev_frame_time != 0:
        # คูณกลับเพื่อให้ได้ FPS จริงของวิดีโอ
        fps = int(1 / (new_frame_time - prev_frame_time)) * (SKIP_FRAMES + 1)
        
        cv2.putText(frame, f"FPS: {fps} (Skipping Mode)", (20, 50), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    prev_frame_time = new_frame_time
    cv2.imshow("High Speed Counting", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()