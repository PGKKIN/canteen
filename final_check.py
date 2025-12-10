import cv2
import time
import torch
from ultralytics import YOLO

print("-" * 50)
print("🔍 เริ่มการตรวจสอบระบบแบบละเอียด...")

# 1. เช็คระดับ PyTorch
if torch.cuda.is_available():
    print(f"✅ PyTorch มองเห็น GPU: {torch.cuda.get_device_name(0)}")
    # สั่งจองที่ไว้ก่อนเลย 500MB เพื่อให้เห็นใน Task Manager ชัวร์ๆ
    dummy_memory = torch.ones(1024, 1024, 100, device='cuda') 
    print(f"💾 VRAM ที่ถูกจองโดย PyTorch: {torch.cuda.memory_allocated() / 1024**2:.2f} MB (ควร > 0)")
else:
    print("❌ PyTorch มองไม่เห็น GPU (จบข่าว)")
    exit()

# 2. โหลดโมเดล
print("⏳ กำลังโหลดโมเดล YOLO...")
model_path = r'D:\my_work\canteen\runs\detect\human_final\weights\best.pt'
model = YOLO(model_path)

# 3. บังคับย้ายเข้าการ์ดจอ (Force Move)
model.to('cuda')

# เช็คว่าโมเดลอยู่ที่ไหน
print(f"🤖 Model Device: {model.device}")
if str(model.device) != 'cuda:0':
    print("⚠️ เตือน: โมเดลยังไม่ได้อยู่ที่ cuda:0 พยายามย้ายอีกครั้ง...")
    model.to('cuda:0')

print("-" * 50)

# 4. เริ่มรันวิดีโอ
video_path = r'D:\my_work\canteen\video.mp4' 
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print(f"❌ Error: หาไฟล์วิดีโอไม่เจอที่ {video_path}")
    print("กรุณาเช็คชื่อไฟล์และโฟลเดอร์ให้ถูกต้อง")
    exit()

print("🚀 เริ่มรันวิดีโอ... (ดู FPS ที่หน้าจอ)")
prev_frame_time = 0

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        print("จบไฟล์วิดีโอ")
        break

    # ลดขนาดภาพ (Resize)
    frame = cv2.resize(frame, (1280, 720))

    # 🔥 จุดสำคัญ: ใส่ device=0 (เป็นตัวเลข) ย้ำไปอีกที
    results = model.track(frame, persist=True, conf=0.25, iou=0.5, verbose=False, device=0)

    # วาดรูปเอง (Manual Drawing)
    if results[0].boxes.id is not None:
        boxes = results[0].boxes.xyxy.cpu().numpy().astype(int)
        ids = results[0].boxes.id.cpu().numpy().astype(int)
        
        for box, id in zip(boxes, ids):
            cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 2)
            cv2.putText(frame, f"#{id}", (box[0], box[1] - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # FPS
    new_frame_time = time.time()
    if prev_frame_time != 0:
        fps = int(1 / (new_frame_time - prev_frame_time))
        cv2.putText(frame, f"FPS: {fps} (GPU Mode)", (20, 50), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    prev_frame_time = new_frame_time
    cv2.imshow("Final GPU Check", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()