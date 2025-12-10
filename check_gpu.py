import sys
print(f"Python Executable: {sys.executable}") 
# บรรทัดบนต้องลงท้ายด้วย .../my_work/python.exe

try:
    import torch
    print(f"Torch Version: {torch.__version__}")
    print(f"GPU Available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU Name: {torch.cuda.get_device_name(0)}")
    else:
        print("❌ ไม่พบ GPU (ต้องลง PyTorch ใหม่)")
except ImportError:
    print("❌ ยังไม่ได้ลง PyTorch (เดี๋ยวเราจะลงกันในขั้นต่อไป)")