import os
import glob

# ⚠️ แก้ Path นี้ให้ตรงกับโฟลเดอร์ dataset ของคุณ
dataset_path = r'D:\my_work\canteen\dataset' 

def convert_labels_to_one_class(folder_path):
    # ค้นหาไฟล์ .txt ทั้งหมดในโฟลเดอร์ labels
    txt_files = glob.glob(os.path.join(folder_path, '**', '*.txt'), recursive=True)
    
    print(f"เจอไฟล์ทั้งหมด {len(txt_files)} ไฟล์ ใน {folder_path}")
    
    count_fixed = 0
    for file_path in txt_files:
        if 'classes.txt' in file_path: continue # ข้ามไฟล์รายชื่อคลาส
        
        with open(file_path, 'r') as f:
            lines = f.readlines()
            
        new_lines = []
        modified = False
        for line in lines:
            parts = line.strip().split()
            if len(parts) > 0:
                class_id = parts[0]
                # ถ้า class_id ไม่ใช่ 0 ให้แก้เป็น 0
                if class_id != '0':
                    parts[0] = '0'
                    line = " ".join(parts) + "\n"
                    modified = True
                new_lines.append(line)
        
        # บันทึกทับไฟล์เดิมเฉพาะที่มีการแก้ไข
        if modified:
            with open(file_path, 'w') as f:
                f.writelines(new_lines)
            count_fixed += 1
            
    print(f"✅ แก้ไขเสร็จสิ้น! แก้ไปทั้งหมด {count_fixed} ไฟล์")

# รันแก้ทั้งโฟลเดอร์ train และ val (หรือ valid)
convert_labels_to_one_class(dataset_path)