import pandas as pd
import numpy as np
from sklearn.utils import shuffle

def generate_complete_horse_dataset(n_samples_per_disease=200):
    """
    สร้างชุดข้อมูลโรคม้าครบ 15 โรคสำหรับ Random Forest
    """
    
    features = {
        'เพศ': ['ผู้', 'เมีย', 'ผู้ตอน'],
        'ช่วงอายุ': ['ลูกม้า', 'ม้าหนุ่มสาว', 'ม้าโต'],
        'พฤติกรรม': ['ปกติ', 'ซึม', 'กินน้อย', 'กระสับกระส่าย', 'ก้าวร้าว'],
        'สภาพแวดล้อม': ['คอกสะอาด', 'คอกสกปรก', 'มีแมลงชุกชุม', 'อากาศแปรปรวน']
    }

    disease_patterns = {
        'โรคพยาธิ': {
            'primary_symptoms': ['น้ำหนักลด', 'ท้องร่วง'],
            'secondary_symptoms': ['เบื่ออาหาร', 'ซึม'],
            'age_preference': 'ลูกม้า',
            'env_preference': 'คอกสกปรก'
        },
        'หลอดอาหารอุดตัน': {
            'primary_symptoms': ['เบื่ออาหาร', 'กลืนลำบาก'],
            'secondary_symptoms': ['น้ำลายไหล', 'ปวดท้อง'],
            'behavior_preference': 'กินน้อย'
        },
        'อาการเสียด': {
            'primary_symptoms': ['ปวดท้อง', 'กระสับกระส่าย'],
            'secondary_symptoms': ['เบื่ออาหาร'],
            'env_preference': 'อากาศแปรปรวน'
        },
        'ภาวะปัสสาวะรั่วออกทางสะดือ': {
            'primary_symptoms': ['ปัสสาวะรั่วทางสะดือ', 'สะดืออักเสบ'],
            'secondary_symptoms': ['ติดเชื้อ'],
            'age_preference': 'ลูกม้า'
        },
        'โรคบาดทะยัก': {
            'primary_symptoms': ['ขากรรไกรแข็ง', 'กล้ามเนื้อกระตุก', 'เกร็ง'],
            'secondary_symptoms': ['ชัก'],
            'env_preference': 'คอกสกปรก'
        },
        'โรคไข้ลงกีบ': {
            'primary_symptoms': ['เจ็บกีบ', 'กีบร้อน', 'เดินกะเผลก'],
            'secondary_symptoms': ['ไข้'],
            'env_preference': 'อากาศแปรปรวน'
        },
        'โรคสมองอักเสบ': {
            'primary_symptoms': ['ชัก', 'เดินโซเซ', 'อัมพาต'],
            'secondary_symptoms': ['ไข้', 'ซึม'],
            'env_preference': 'มีแมลงชุกชุม'
        },
        'โรคโลหิตจางติดต่อ': {
            'primary_symptoms': ['ซึม', 'น้ำหนักลด', 'บวมน้ำ'],
            'secondary_symptoms': ['ไข้'],
            'env_preference': 'มีแมลงชุกชุม'
        },
        'โรคเซอร์รา': {
            'primary_symptoms': ['อ่อนแรง', 'เดินไม่มั่นคง', 'ตาอักเสบ'],
            'secondary_symptoms': ['ไข้', 'บวมน้ำ'],
            'env_preference': 'มีแมลงชุกชุม'
        },
        'โรคมงคล่อธรรมดา': {
            'primary_symptoms': ['ต่อมน้ำเหลืองบวม', 'กลืนลำบาก', 'คอบวม'],
            'secondary_symptoms': ['ไข้', 'ไอ'],
            'age_preference': 'ลูกม้า'
        },
        'โรคไข้หวัดใหญ่': {
            'primary_symptoms': ['ไข้', 'ไอ', 'น้ำมูกไหล'],
            'secondary_symptoms': ['เบื่ออาหาร', 'ซึม'],
            'env_preference': 'อากาศแปรปรวน'
        },
        'โรคจมูกอักเสบจากไวรัส': {
            'primary_symptoms': ['น้ำมูกเขียว', 'แท้ง', 'หายใจลำบาก'],
            'secondary_symptoms': ['ไข้'],
            'sex_preference': 'เมีย'
        },
        'โรคเออร์ลิชิโอสิส': {
            'primary_symptoms': ['ไข้', 'ขาบวม', 'เดินผิดปกติ'],
            'secondary_symptoms': ['ซึม'],
            'env_preference': 'มีแมลงชุกชุม'
        },
        'โรคฉี่หนู': {
            'primary_symptoms': ['ดีซ่าน', 'ตาอักเสบ', 'ปัสสาวะผิดปกติ'],
            'secondary_symptoms': ['ไข้'],
            'env_preference': 'คอกสกปรก'
        },
        'โรคซัลโมเนลโลสิส': {
            'primary_symptoms': ['ท้องร่วง', 'ไข้'],
            'secondary_symptoms': ['เบื่ออาหาร', 'ซึม'],
            'age_preference': 'ลูกม้า'
        }
    }

    # รวบรวมอาการทั้งหมด
    all_symptoms = set()
    for pattern in disease_patterns.values():
        all_symptoms.update(pattern['primary_symptoms'])
        all_symptoms.update(pattern['secondary_symptoms'])

    data = []
    # สร้างข้อมูลสำหรับแต่ละโรค
    for disease, pattern in disease_patterns.items():
        for _ in range(n_samples_per_disease):
            record = {
                'เพศ': pattern.get('sex_preference', np.random.choice(features['เพศ'])),
                'ช่วงอายุ': pattern.get('age_preference', np.random.choice(features['ช่วงอายุ'])),
                'พฤติกรรม': pattern.get('behavior_preference', np.random.choice(features['พฤติกรรม'])),
                'สภาพแวดล้อม': pattern.get('env_preference', np.random.choice(features['สภาพแวดล้อม']))
            }
            
            # เพิ่มอาการทั้งหมดเป็น 0
            for symptom in all_symptoms:
                record[symptom] = 0
            
            # ใส่อาการหลัก
            for symptom in pattern['primary_symptoms']:
                record[symptom] = np.random.choice([0, 1], p=[0.1, 0.9])  # 90% chance of having primary symptoms
            
            # ใส่อาการรอง
            for symptom in pattern['secondary_symptoms']:
                record[symptom] = np.random.choice([0, 1], p=[0.3, 0.7])  # 70% chance of having secondary symptoms
            
            # เพิ่ม noise
            for symptom in np.random.choice(list(all_symptoms), size=2):
                if symptom not in pattern['primary_symptoms'] + pattern['secondary_symptoms']:
                    record[symptom] = np.random.choice([0, 1], p=[0.8, 0.2])  # 20% chance of random symptoms
            
            record['โรค'] = disease
            data.append(record)
    
    # สร้าง DataFrame และสลับข้อมูล
    df = pd.DataFrame(data)
    df = shuffle(df, random_state=42)
    
    return df

# สร้างและบันทึกข้อมูล
df = generate_complete_horse_dataset(200)  # 200 ตัวอย่างต่อโรค
df.to_csv('complete_horse_disease_dataset.csv', index=False, encoding='utf-8-sig')

# แสดงสถิติของข้อมูล
print("ขนาดชุดข้อมูล:", df.shape)
print("\nการกระจายของโรค:")
print(df['โรค'].value_counts())
print("\nรายการอาการทั้งหมด:")
symptoms = [col for col in df.columns if col not in ['เพศ', 'ช่วงอายุ', 'พฤติกรรม', 'สภาพแวดล้อม', 'โรค']]
print(symptoms)