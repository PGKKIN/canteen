import onnxruntime as ort
import sys

print(f"ONNX Runtime Version: {ort.__version__}")
print(f"Available Providers: {ort.get_available_providers()}")

# เช็คว่ามี QNN หรือ SNPE ไหม
if 'QNNExecutionProvider' in ort.get_available_providers():
    print("✅ พบ QNN Execution Provider (NPU พร้อมใช้งาน!)")
elif 'SNPEExecutionProvider' in ort.get_available_providers():
    print("✅ พบ SNPE Execution Provider (Legacy NPU พร้อมใช้งาน!)")
else:
    print("❌ ไม่พบตัวเร่งความเร็ว NPU (QNN/SNPE)")
    print("⚠️ ตอนนี้ระบบกำลังใช้: " + str(ort.get_available_providers()[0]))
    print("👉 คุณอาจต้องลง 'onnxruntime-qnn' หรือไลบรารีจาก Qualcomm/Radxa")

# เช็คเรื่อง GPU Driver (เผื่อใช้ GPU ได้)
try:
    with open("/sys/class/drm/card0/device/vendor", "r") as f:
        print(f"GPU Vendor ID: {f.read().strip()}")
except Exception as e:
    print(f"⚠️ อ่านค่า GPU ไม่ได้ (อาจจะไม่มีสิทธิ์ หรือ Driver ไม่ครบ): {e}")
