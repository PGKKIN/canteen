import cv2
import time
import os
from ultralytics import YOLO

# 1. ตั้งค่า Path ให้เป็น Relative (ใช้ได้ทั้ง Windows/Linux)
# ค้นหาไฟล์โมเดล .onnx ก่อน ถ้าไม่มีค่อยหา .pt
onnx_path = 'runs/detect/human_final/weights/best.onnx'
pt_path = 'runs/detect/human_final/weights/best.pt'

if os.path.exists(onnx_path):
    model_file = onnx_path
    print(f"✅ พบโมเดล ONNX: {model_file} (แนะนำสำหรับ Radxa)")
elif os.path.exists(pt_path):
    model_file = pt_path
    print(f"⚠️ ไม่พบ ONNX, ใช้โมเดล PT แทน: {model_file}")
else:
    print("❌ ไม่พบไฟล์โมเดลเลย! กรุณาตรวจสอบ")
    exit()

# 2. โหลดโมเดล
try:
    # task='detect' ช่วยให้มั่นใจว่าโหลดถูกโหมด
    model = YOLO(model_file, task='detect') 
except Exception as e:
    print(f"❌ Error Loading Model: {e}")
    exit()

# 3. เตรียมวิดีโอ (ใช้ path แบบ relative)
video_file = 'video.mp4'
if not os.path.exists(video_file):
    print(f"❌ ไม่พบไฟล์วิดีโอ: {video_file}")
    print("จะลองเปิดกล้อง (Webcam) แทน...")
    cap = cv2.VideoCapture(0)
else:
    print(f"🎥 กำลังรันไฟล์วิดีโอ: {video_file}")
    cap = cv2.VideoCapture(video_file)

if not cap.isOpened():
    print("❌ ไม่สามารถเปิดวิดีโอหรือกล้องได้")
    exit()

# 4. ลูปแสดงผล
print("🚀 เริ่มการทำงาน... กด 'q' เพื่อออก")
prev_frame_time = 0

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        print("จบไฟล์วิดีโอ (Loop)")
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0) # วนลูปวิดีโอถ้าจบ
        continue

    # ย่อภาพหน่อยถ้าระบบช้า (Radxa ควรไหวที่ 640-1280)
    # frame = cv2.resize(frame, (640, 360)) 

    # Run Inference
    # device='0' หรือ 'cpu' หรือปล่อยว่างให้ auto
    results = model.predict(frame, conf=0.3, iou=0.5, verbose=False)

    # วาดผลลัพธ์
    annotated_frame = results[0].plot()

    # FPS Calculation
    new_frame_time = time.time()
    fps = 1 / (new_frame_time - prev_frame_time) if prev_frame_time > 0 else 0
    prev_frame_time = new_frame_time

    # แสดง FPS
    cv2.putText(annotated_frame, f"FPS: {int(fps)}", (20, 40), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Show (ระวัง: ถ้า SSH มาแล้วไม่มี X11 อาจจะ Error ตรงนี้)
    try:
        cv2.imshow("Radxa AI Inspection", annotated_frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    except Exception as e:
        print(f"Frame processed. FPS: {int(fps)} (No Display)")

cap.release()
cv2.destroyAllWindows()
