import pandas as pd
import joblib
import os

def predict_disease(input_case):
    """ทำนายโรคม้าจากอาการที่พบ"""
    try:
        # โหลดโมเดล
        if not os.path.exists('horse_disease_model.joblib'):
            print("ไม่พบไฟล์โมเดล")
            return None
            
        model_data = joblib.load('horse_disease_model.joblib')
        model = model_data['model']
        label_encoders = model_data['label_encoders']
        
        # ดึง feature names จากโมเดล
        feature_names = model.feature_names_in_
        
        # สร้าง DataFrame ด้วย features ที่ถูกต้อง
        input_processed = pd.DataFrame(columns=feature_names)
        input_processed.loc[0] = 0  # กำหนดค่าเริ่มต้นเป็น 0
        
        # ใส่ค่าจาก input_case
        for feature, value in input_case.items():
            if feature in feature_names:
                if feature in label_encoders:
                    input_processed.loc[0, feature] = label_encoders[feature].transform([value])[0]
                else:
                    input_processed.loc[0, feature] = value
            else:
                print(f"ข้อมูล '{feature}' ไม่ได้ใช้ในโมเดล")
        
        # ทำนาย
        predictions = model.predict_proba(input_processed)[0]
        diseases = model.classes_
        
        # หาโรคที่มีความน่าจะเป็นสูงสุด
        max_prob_index = predictions.argmax()
        predicted_disease = diseases[max_prob_index]
        probability = predictions[max_prob_index]
        
        print(f"\nผลการวินิจฉัย: {predicted_disease} (ความน่าจะเป็น {probability:.2%})")
            
        return predicted_disease, probability
        
    except Exception as e:
        print(f"เกิดข้อผิดพลาดในการทำนาย: {str(e)}")
        return None

# ทดสอบการทำนาย
if __name__ == "__main__":
    print("กำลังโหลดโมเดล...")
    model_data = joblib.load('horse_disease_model.joblib')
    print("Features ที่ต้องใช้:")
    for feature in model_data['model'].feature_names_in_:
        print(f"- {feature}")
        
    # ตัวอย่างข้อมูลสำหรับทำนาย
    test_case = {
        'เพศ': 'เมีย',
        'ช่วงอายุ': 'ลูกม้า',
        'พฤติกรรม': 'ซึม',
        'สภาพแวดล้อม': 'คอกสกปรก',
        'ไข้': 1,
        'เบื่ออาหาร': 1,
        'น้ำหนักลด': 1,
        'ท้องร่วง': 1,
        'ซึม': 1,
        'ขาบวม': 1,
        'ติดเชื้อ': 1,
        'เดินผิดปกติ': 1
    }
    
    print("\nทดสอบการทำนาย...")
    result = predict_disease(test_case)