import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm.notebook import tqdm
import time

class HorseDiseaseClassifier:
    def __init__(self):
        self.model = None
        self.label_encoders = {}
        self.feature_names = None
        self.categorical_features = ['เพศ', 'ช่วงอายุ', 'พฤติกรรม', 'สภาพแวดล้อม']
        
    def preprocess_data(self, df):
        """แปลงข้อมูลให้เหมาะสมกับการเทรน"""
        print("กำลังเตรียมข้อมูล... 0%")
        df_processed = df.copy()
        
        # Encode categorical variables
        for i, feature in enumerate(self.categorical_features):
            self.label_encoders[feature] = LabelEncoder()
            df_processed[feature] = self.label_encoders[feature].fit_transform(df_processed[feature])
            progress = (i + 1) / len(self.categorical_features) * 100
            print(f"กำลังเตรียมข้อมูล... {progress:.0f}%")
            
        return df_processed
    
    def train(self, df, tune_hyperparameters=True):
        """เทรนโมเดลพร้อมแสดงความคืบหน้า"""
        print("เริ่มกระบวนการเทรนโมเดล...")
        start_time = time.time()
        
        # Preprocess data
        print("\n1. การเตรียมข้อมูล")
        df_processed = self.preprocess_data(df)
        
        # แยก features และ target
        X = df_processed.drop('โรค', axis=1)
        y = df_processed['โรค']
        self.feature_names = X.columns
        
        print("\n2. การแบ่งข้อมูล")
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        print(f"จำนวนข้อมูลเทรน: {len(X_train)}")
        print(f"จำนวนข้อมูลทดสอบ: {len(X_test)}")
        
        if tune_hyperparameters:
            print("\n3. การหาค่าพารามิเตอร์ที่เหมาะสม")
            param_grid = {
                'n_estimators': [100, 200, 300],
                'max_depth': [10, 20, 30, None],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4],
                'max_features': ['sqrt', 'log2', None],
                'class_weight': ['balanced', 'balanced_subsample', None]
            }
            
            total_combinations = (len(param_grid['n_estimators']) * 
                               len(param_grid['max_depth']) * 
                               len(param_grid['min_samples_split']) * 
                               len(param_grid['min_samples_leaf']) * 
                               len(param_grid['max_features']) * 
                               len(param_grid['class_weight']))
            
            print(f"จำนวน combinations ที่จะทดสอบ: {total_combinations}")
            
            rf = RandomForestClassifier(random_state=42)
            grid_search = GridSearchCV(
                rf, param_grid, 
                cv=5, 
                n_jobs=-1,
                verbose=2,
                scoring='f1_weighted'
            )
            
            print("\nเริ่ม Grid Search...")
            grid_search.fit(X_train, y_train)
            
            self.model = grid_search.best_estimator_
            print("\nผลการ Grid Search:")
            print(f"Best parameters: {grid_search.best_params_}")
            print(f"Best CV score: {grid_search.best_score_:.4f}")
        else:
            print("\n3. การเทรนโมเดลด้วยค่าพารามิเตอร์ที่กำหนด")
            self.model = RandomForestClassifier(
                n_estimators=200,
                max_depth=20,
                min_samples_split=5,
                min_samples_leaf=2,
                max_features='sqrt',
                class_weight='balanced',
                random_state=42,
                verbose=2
            )
            self.model.fit(X_train, y_train)
        
        print("\n4. การประเมินผลโมเดล")
        y_pred = self.model.predict(X_test)
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred))
        
        total_time = time.time() - start_time
        print(f"\nเวลาที่ใช้ในการเทรนทั้งหมด: {total_time:.2f} วินาที")
        
        print("\n5. การสร้างกราฟแสดงผล")
        self.plot_confusion_matrix(y_test, y_pred)
        self.plot_feature_importance()
        
        return self.model

    def predict_proba(self, input_data):
        """ทำนายความน่าจะเป็นของแต่ละโรค"""
        if isinstance(input_data, dict):
            input_data = pd.DataFrame([input_data])
        
        # Preprocess input
        input_processed = input_data.copy()
        for feature in self.categorical_features:
            if feature in input_processed.columns:
                input_processed[feature] = self.label_encoders[feature].transform(input_processed[feature])
        
        # Ensure all features are present
        for feature in self.feature_names:
            if feature not in input_processed.columns:
                input_processed[feature] = 0
        
        # Get predictions and probabilities
        probas = self.model.predict_proba(input_processed[self.feature_names])
        disease_probs = dict(zip(self.model.classes_, probas[0]))
        
        # Sort by probability
        sorted_probs = {k: v for k, v in sorted(
            disease_probs.items(), 
            key=lambda item: item[1], 
            reverse=True
        )}
        
        return sorted_probs

    def plot_confusion_matrix(self, y_true, y_pred):
        """Plot confusion matrix with better visualization"""
        plt.figure(figsize=(15, 10))
        cm = confusion_matrix(y_true, y_pred)
        sns.heatmap(
            cm, 
            annot=True, 
            fmt='d',
            cmap='YlOrRd',
            xticklabels=self.model.classes_,
            yticklabels=self.model.classes_
        )
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.xticks(rotation=45)
        plt.yticks(rotation=45)
        plt.tight_layout()
        plt.show()

    def plot_feature_importance(self):
        """Plot feature importance with better visualization"""
        importances = pd.DataFrame({
            'feature': self.feature_names,
            'importance': self.model.feature_importances_
        })
        importances = importances.sort_values('importance', ascending=False)
        
        plt.figure(figsize=(12, 6))
        sns.barplot(
            data=importances.head(20),
            x='importance',
            y='feature',
            palette='viridis'
        )
        plt.title('Top 20 Most Important Features')
        plt.xlabel('Feature Importance')
        plt.tight_layout()
        plt.show()

    def get_model_summary(self):
        """แสดงสรุปข้อมูลโมเดล"""
        if self.model is None:
            return "โมเดลยังไม่ได้รับการเทรน"
        
        summary = {
            'จำนวน Trees': self.model.n_estimators,
            'Max Depth': self.model.max_depth,
            'Min Samples Split': self.model.min_samples_split,
            'Min Samples Leaf': self.model.min_samples_leaf,
            'Max Features': self.model.max_features,
            'Class Weight': self.model.class_weight,
            'จำนวน Features': len(self.feature_names),
            'Features ที่ใช้': list(self.feature_names)
        }
        
        return pd.Series(summary)

    def save_model(self, filename):
        """บันทึกโมเดลและข้อมูลที่จำเป็น"""
        import joblib
        model_data = {
            'model': self.model,
            'label_encoders': self.label_encoders,
            'feature_names': self.feature_names,
            'categorical_features': self.categorical_features
        }
        joblib.dump(model_data, filename)
        print(f"บันทึกโมเดลไปที่: {filename}")

    @classmethod
    def load_model(cls, filename):
        """โหลดโมเดลที่บันทึกไว้"""
        import joblib
        model_data = joblib.load(filename)
        
        classifier = cls()
        classifier.model = model_data['model']
        classifier.label_encoders = model_data['label_encoders']
        classifier.feature_names = model_data['feature_names']
        classifier.categorical_features = model_data['categorical_features']
        
        return classifier

# ตัวอย่างการใช้งาน
if __name__ == "__main__":
    print("กำลังโหลดข้อมูล...")
    df = pd.read_csv('complete_horse_disease_dataset.csv')
    print(f"จำนวนข้อมูลทั้งหมด: {len(df)}")
    print(f"จำนวนโรคทั้งหมด: {df['โรค'].nunique()}")
    print("\nการกระจายของข้อมูล:")
    print(df['โรค'].value_counts())
    
    # สร้างและเทรนโมเดล
    classifier = HorseDiseaseClassifier()
    classifier.train(df, tune_hyperparameters=True)
    
    # บันทึกโมเดล
    classifier.save_model('horse_disease_model.joblib')
    
    # ตัวอย่างการทำนาย
    test_case = {
        'เพศ': 'เมีย',
        'ช่วงอายุ': 'ลูกม้า',
        'พฤติกรรม': 'ซึม',
        'สภาพแวดล้อม': 'คอกสกปรก',
        'ไข้': 1,
        'เบื่ออาหาร': 1,
        'น้ำหนักลด': 1,
        'ท้องร่วง': 1
    }
    
    predictions = classifier.predict_proba(test_case)
    print("\nผลการทำนาย 3 อันดับแรก:")
    for i, (disease, prob) in enumerate(list(predictions.items())[:3], 1):
        print(f"{i}. {disease}: {prob:.2%}")