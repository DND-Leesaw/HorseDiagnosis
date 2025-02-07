import pandas as pd
import json

def load_diseases_from_csv(csv_path='horse_diseases.csv', json_path='diseases_data.json'):
    try:
        # อ่านข้อมูลจาก CSV
        df = pd.read_csv(csv_path, encoding='utf-8')
        
        # แปลงข้อมูลเป็น dictionary
        diseases_dict = {
            row['ชื่อโรค (ไทย)']: {
                "ชื่อโรค (ไทย)": row['ชื่อโรค (ไทย)'],
                "ชื่อโรค (อังกฤษ)": row['ชื่อโรค (อังกฤษ)'],
                "อาการ": row['อาการ'],
                "การควบคุมและป้องกัน": row['การควบคุมและป้องกัน'],
                "ระดับความรุนแรง": row['ระดับความรุนแรง']
            }
            for _, row in df.iterrows()
        }
        
        # บันทึกข้อมูลลง JSON file
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(diseases_dict, f, ensure_ascii=False, indent=4)
        
        print(f"Saved diseases data to {json_path}")
        return diseases_dict
    except Exception as e:
        print(f"Error loading diseases data: {str(e)}")
        return {}

# เรียกใช้ฟังก์ชัน
load_diseases_from_csv()