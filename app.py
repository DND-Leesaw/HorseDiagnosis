from flask import Flask, render_template, request, redirect, url_for, flash, jsonify, session
from flask_wtf.csrf import CSRFProtect, CSRFError
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
from flask_wtf import FlaskForm
from wtforms import StringField, TextAreaField, SelectField
from wtforms.validators import DataRequired
from functools import wraps, lru_cache
import pandas as pd
import numpy as np
import joblib
import os
import logging
from datetime import datetime, timedelta
import json
import shutil
from werkzeug.security import generate_password_hash, check_password_hash
from werkzeug.utils import secure_filename
import hashlib
import hmac
import traceback

# Initialize Flask app
app = Flask(__name__)

# Load config
class Config:
    SECRET_KEY = os.getenv('SECRET_KEY', 'your-secret-key')
    WTF_CSRF_SECRET_KEY = os.getenv('WTF_CSRF_SECRET_KEY', 'your-csrf-secret-key')
    SESSION_TYPE = 'filesystem'
    PERMANENT_SESSION_LIFETIME = timedelta(minutes=int(os.getenv('SESSION_LIFETIME', 30)))
    MAX_CONTENT_LENGTH = 100 * 1024 * 1024  # 100MB max file size
    UPLOAD_FOLDER = os.getenv('UPLOAD_FOLDER', 'uploads')
    MODEL_FOLDER = 'models'
    BACKUP_FOLDER = 'backups'
    LOG_FOLDER = 'logs'
    DEBUG = os.getenv('FLASK_DEBUG', 'False').lower() in ('true', '1', 't')
    ENV = os.getenv('FLASK_ENV', 'production')
    GITHUB_SECRET = os.getenv('GITHUB_SECRET', None)

app.config.from_object(Config)

# Initialize extensions
csrf = CSRFProtect(app)
limiter = Limiter(app=app, key_func=get_remote_address)

# Constants
ADMIN_USERNAME = os.getenv('ADMIN_USERNAME', 'admin')
ADMIN_PASSWORD_HASH = os.getenv('ADMIN_PASSWORD_HASH', generate_password_hash('admin'))
MODEL_FILENAME = 'horse_disease_model.joblib'
MODEL_PATH = os.path.join(Config.MODEL_FOLDER, MODEL_FILENAME)
ALLOWED_EXTENSIONS = {'joblib', 'pkl'}

# Create required folders
for folder in [Config.MODEL_FOLDER, 'static', Config.BACKUP_FOLDER, Config.UPLOAD_FOLDER, Config.LOG_FOLDER]:
    os.makedirs(folder, exist_ok=True)

# Utility Classes
class ModelManager:
    def __init__(self, model_folder, backup_folder):
        self.model_folder = model_folder
        self.backup_folder = backup_folder
        
    def list_models(self):
        models = []
        if os.path.exists(self.model_folder):
            for filename in os.listdir(self.model_folder):
                if filename.endswith(('.joblib', '.pkl')):
                    file_path = os.path.join(self.model_folder, filename)
                    models.append({
                        'model_name': filename,
                        'upload_date': datetime.fromtimestamp(
                            os.path.getctime(file_path)
                        ).strftime('%Y-%m-%d %H:%M:%S'),
                        'status': 'active' if filename == os.path.basename(MODEL_PATH) else 'inactive',
                        'size': f"{os.path.getsize(file_path) / (1024*1024):.2f} MB"
                    })
        return sorted(models, key=lambda x: x['upload_date'], reverse=True)

    def validate_model(self, model_path):
        try:
            model_data = joblib.load(model_path)
            required_keys = ['model', 'label_encoders', 'feature_names']
            return all(key in model_data for key in required_keys)
        except Exception as e:
            logger.error(f"Model validation failed: {str(e)}")
            return False

class FileManager:
    def __init__(self, upload_folder, backup_folder):
        self.upload_folder = upload_folder
        self.backup_folder = backup_folder
        
    def save_file(self, file, filename, subfolder=None):
        if subfolder:
            save_path = os.path.join(self.upload_folder, subfolder)
            os.makedirs(save_path, exist_ok=True)
        else:
            save_path = self.upload_folder
            
        temp_name = f'temp_{filename}'
        temp_path = os.path.join(save_path, temp_name)
        final_path = os.path.join(save_path, filename)
        
        try:
            file.save(temp_path)
            os.replace(temp_path, final_path)
            return final_path
        except Exception as e:
            if os.path.exists(temp_path):
                os.remove(temp_path)
            raise e

    def get_folder_size(self, folder):
        total_size = 0
        for dirpath, dirnames, filenames in os.walk(folder):
            for filename in filenames:
                filepath = os.path.join(dirpath, filename)
                total_size += os.path.getsize(filepath)
        return total_size

class ActivityLogger:
    def __init__(self, log_folder):
        self.log_folder = log_folder
        self.setup_logging()
        
    def setup_logging(self):
        os.makedirs(self.log_folder, exist_ok=True)
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(
                    os.path.join(self.log_folder, 'app.log'),
                    encoding='utf-8'
                ),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger('app_activity')

    def log_activity(self, user, action, details=None):
        log_entry = {
            'timestamp': datetime.now().isoformat(),
            'user': user,
            'action': action,
            'details': details,
            'ip': request.remote_addr if request else None
        }
        
        log_file = os.path.join(
            self.log_folder,
            f'activity_{datetime.now().strftime("%Y%m")}.json'
        )
        
        try:
            existing_logs = []
            if os.path.exists(log_file):
                with open(log_file, 'r', encoding='utf-8') as f:
                    existing_logs = json.load(f)
                    
            existing_logs.append(log_entry)
            
            with open(log_file, 'w', encoding='utf-8') as f:
                json.dump(existing_logs, f, ensure_ascii=False, indent=2)
                
            self.logger.info(f"Activity: {action}, User: {user}, Details: {details}")
            
        except Exception as e:
            self.logger.error(f"Failed to log activity: {str(e)}")

# Initialize managers
model_manager = ModelManager(Config.MODEL_FOLDER, Config.BACKUP_FOLDER)
file_manager = FileManager(Config.UPLOAD_FOLDER, Config.BACKUP_FOLDER)
activity_logger = ActivityLogger(Config.LOG_FOLDER)
logger = logging.getLogger(__name__)

# Forms
class DiseaseForm(FlaskForm):
    thai_name = StringField('ชื่อโรค (ไทย)', validators=[DataRequired()])
    eng_name = StringField('ชื่อโรค (อังกฤษ)', validators=[DataRequired()])
    symptoms = TextAreaField('อาการ', validators=[DataRequired()])
    prevention = TextAreaField('การควบคุมและป้องกัน', validators=[DataRequired()])
    severity = SelectField('ระดับความรุนแรง',
                         choices=[('ต่ำ', 'ต่ำ'), ('กลาง', 'กลาง'), ('สูง', 'สูง')],
                         validators=[DataRequired()])

# Utility Functions
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@lru_cache(maxsize=1)
def load_model(model_path=MODEL_PATH):
    if os.path.exists(model_path):
        try:
            model_data = joblib.load(model_path)
            return model_data['model'], model_data['label_encoders'], model_data['feature_names']
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}", exc_info=True)
    return None, None, None

def load_diseases_data():
    try:
        if not os.path.exists('diseases_data.json'):
            with open('diseases_data.json', 'w', encoding='utf-8') as f:
                json.dump({}, f, ensure_ascii=False, indent=4)
        with open('diseases_data.json', 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"Error loading diseases data: {str(e)}")
        return {}

def process_input_data(input_data, label_encoders, feature_names):
    try:
        processed_data = []
        for feature in feature_names:
            value = input_data.get(feature)
            if feature in label_encoders:
                value = label_encoders[feature].transform([value])[0]
            processed_data.append(value)
        return processed_data
    except Exception as e:
        logger.error(f"Error processing input data: {str(e)}")
        raise e

# Decorators
def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if not session.get('logged_in'):
            flash('กรุณาเข้าสู่ระบบก่อนเข้าใช้งาน', 'error')
            return redirect(url_for('admin_login'))
        return f(*args, **kwargs)
    return decorated_function

# Load initial data
model, label_encoders, feature_names = load_model()
diseases_data = load_diseases_data()

# Routes
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/admin/login', methods=['GET', 'POST'])
@limiter.limit("5 per minute")
def admin_login():
    if request.method == 'POST':
        try:
            username = request.form.get('username')
            password = request.form.get('password')
            
            if not username or not password:
                flash('กรุณากรอกข้อมูลให้ครบถ้วน', 'error')
                return render_template('admin_login.html')
            
            if username == ADMIN_USERNAME and check_password_hash(ADMIN_PASSWORD_HASH, password):
                session['logged_in'] = True
                session['username'] = username
                session.permanent = True
                
                activity_logger.log_activity(
                    username,
                    'login',
                    {'ip': request.remote_addr}
                )
                
                flash('เข้าสู่ระบบสำเร็จ', 'success')
                return redirect(url_for('admin_dashboard'))
            else:
                logger.warning(f'Login failed for user: {username}')
                flash('ชื่อผู้ใช้หรือรหัสผ่านไม่ถูกต้อง', 'error')
                
        except Exception as e:
            logger.error(f"Error in login: {str(e)}")
            flash('เกิดข้อผิดพลาดในการเข้าสู่ระบบ', 'error')
    
    return render_template('admin_login.html')

@app.route('/admin/logout')
def admin_logout():
    if session.get('username'):
        activity_logger.log_activity(
            session['username'],
            'logout'
        )
    session.clear()
    flash('ออกจากระบบสำเร็จ', 'success')
    return redirect(url_for('home'))

@app.route('/admin/dashboard', methods=['GET', 'POST'])
@login_required
def admin_dashboard():
    try:
        form = DiseaseForm()
        
        if request.method == 'POST':
            action = request.form.get('action')
            
            try:
                if action == 'add_model':
                    if 'model_file' not in request.files:
                        raise ValueError('กรุณาเลือกไฟล์โมเดล')
                    
                    file = request.files['model_file']
                    if not file or not file.filename:
                        raise ValueError('ไม่พบไฟล์ที่อัพโหลด')
                        
                    if not allowed_file(file.filename):
                        raise ValueError('รองรับเฉพาะไฟล์ .joblib และ .pkl เท่านั้น')
                        
                    filename = secure_filename(file.filename)
                    filepath = file_manager.save_file(file, filename, 'models')
                    
                    if not model_manager.validate_model(filepath):
                        os.remove(filepath)
                        raise ValueError('ไฟล์โมเดลไม่ถูกต้อง')
                        
                    activity_logger.log_activity(
                        session['username'],
                        'upload_model',
                        {'model_name': filename}
                    )
                    flash('อัพโหลดโมเดลสำเร็จ', 'success')
                    
                elif action == 'set_active':
                    model_name = request.form.get('model_name')
                    model_path = os.path.join(Config.MODEL_FOLDER, model_name)
                    
                    global model, label_encoders, feature_names, MODEL_PATH
                    model, label_encoders, feature_names = load_model(model_path)
                    
                    if model is None:
                        raise ValueError('ไม่สามารถโหลดโมเดลได้')
                        
                    MODEL_PATH = model_path
                    activity_logger.log_activity(
                        session['username'],
                        'activate_model',
                        {'model_name': model_name}
                    )
                    flash('เปลี่ยนโมเดลที่ใช้งานสำเร็จ', 'success')
                    
                elif action == 'delete_model':
                    model_name = request.form.get('model_name')
                    model_path = os.path.join(Config.MODEL_FOLDER, model_name)
                    
                     if os.path.exists(model_path):
                      # สำรองข้อมูลก่อนลบ
                      backup_name = f"backup_{model_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                         shutil.copy2(model_path, os.path.join(Config.BACKUP_FOLDER, backup_name))

                    
                    os.remove(model_path)
                    activity_logger.log_activity(
                        session['username'],
                        'delete_model',
                        {'model_name': model_name}
                    )
                    flash('ลบโมเดลสำเร็จ', 'success')
                
                elif action == 'add_disease':
                    if form.validate_on_submit():
                        thai_name = form.thai_name.data
                        eng_name = form.eng_name.data
                        symptoms = form.symptoms.data
                        prevention = form.prevention.data
                        severity = form.severity.data
                        
                        if thai_name in diseases_data:
                            raise ValueError('มีข้อมูลโรคนี้อยู่แล้ว')
                            
                        diseases_data[thai_name] = {
                            'ชื่อโรค (ไทย)': thai_name,
                            'ชื่อโรค (อังกฤษ)': eng_name,
                            'อาการ': symptoms,
                            'การควบคุมและป้องกัน': prevention,
                            'ระดับความรุนแรง': severity,
                            'last_updated': datetime.now().isoformat(),
                            'updated_by': session['username']
                        }
                        
                        with open('diseases_data.json', 'w', encoding='utf-8') as f:
                            json.dump(diseases_data, f, ensure_ascii=False, indent=4)
                            
                        activity_logger.log_activity(
                            session['username'],
                            'add_disease',
                            {'disease_name': thai_name}
                        )
                        flash('เพิ่มข้อมูลโรคสำเร็จ', 'success')
                    else:
                        for field, errors in form.errors.items():
                            for error in errors:
                                flash(f"{getattr(form, field).label.text}: {error}", 'error')
                                
                elif action == 'edit_disease':
                    disease_id = request.form.get('disease_id')
                    thai_name = request.form.get('thai_name')
                    eng_name = request.form.get('eng_name')
                    symptoms = request.form.get('symptoms')
                    prevention = request.form.get('prevention')
                    severity = request.form.get('severity')
                    
                    if not all([disease_id, thai_name, eng_name, symptoms, prevention, severity]):
                        raise ValueError('กรุณากรอกข้อมูลให้ครบถ้วน')
                        
                    disease_keys = list(diseases_data.keys())
                    old_disease_name = disease_keys[int(disease_id) - 1]
                    
                    # สำรองข้อมูลก่อนแก้ไข
                    backup = {
                        'edited_at': datetime.now().isoformat(),
                        'edited_by': session['username'],
                        'old_data': diseases_data[old_disease_name]
                    }
                    
                    backup_filename = f'edited_disease_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
                    with open(os.path.join(Config.BACKUP_FOLDER, backup_filename), 'w', encoding='utf-8') as f:
                        json.dump(backup, f, ensure_ascii=False, indent=4)
                    
                    # อัปเดตข้อมูล
                    if old_disease_name != thai_name:
                        del diseases_data[old_disease_name]
                        
                    diseases_data[thai_name] = {
                        'ชื่อโรค (ไทย)': thai_name,
                        'ชื่อโรค (อังกฤษ)': eng_name,
                        'อาการ': symptoms,
                        'การควบคุมและป้องกัน': prevention,
                        'ระดับความรุนแรง': severity,
                        'last_updated': datetime.now().isoformat(),
                        'updated_by': session['username']
                    }
                    
                    with open('diseases_data.json', 'w', encoding='utf-8') as f:
                        json.dump(diseases_data, f, ensure_ascii=False, indent=4)
                        
                    activity_logger.log_activity(
                        session['username'],
                        'edit_disease',
                        {
                            'old_name': old_disease_name,
                            'new_name': thai_name
                        }
                    )
                    flash('แก้ไขข้อมูลโรคสำเร็จ', 'success')
                    
                elif action == 'delete_disease':
                    disease_id = request.form.get('disease_id')
                    disease_keys = list(diseases_data.keys())
                    disease_name = disease_keys[int(disease_id) - 1]
                    
                    # สำรองข้อมูลก่อนลบ
                    backup = {
                        'deleted_at': datetime.now().isoformat(),
                        'deleted_by': session['username'],
                        'disease_data': diseases_data[disease_name]
                    }
                    
                    backup_filename = f'deleted_disease_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
                    with open(os.path.join(Config.BACKUP_FOLDER, backup_filename), 'w', encoding='utf-8') as f:
                        json.dump(backup, f, ensure_ascii=False, indent=4)
                    
                    del diseases_data[disease_name]
                    with open('diseases_data.json', 'w', encoding='utf-8') as f:
                        json.dump(diseases_data, f, ensure_ascii=False, indent=4)
                        
                    activity_logger.log_activity(
                        session['username'],
                        'delete_disease',
                        {'disease_name': disease_name}
                    )
                    flash('ลบข้อมูลโรคสำเร็จ', 'success')
                    
            except Exception as e:
                logger.error(f"Error in dashboard action {action}: {str(e)}")
                flash(str(e), 'error')
                
            return redirect(url_for('admin_dashboard'))
            
        # GET request
        models = model_manager.list_models()
        stats = {
            'total_models': len(models),
            'active_models': len([m for m in models if m['status'] == 'active']),
            'total_diseases': len(diseases_data),
            'disk_usage': {
                'models': file_manager.get_folder_size(Config.MODEL_FOLDER) / (1024*1024),
                'backups': file_manager.get_folder_size(Config.BACKUP_FOLDER) / (1024*1024)
            }
        }
        
        return render_template('admin_dashboard.html',
                             models=models,
                             diseases=diseases_data,
                             stats=stats,
                             form=form)
                             
    except Exception as e:
        logger.error(f"Error loading dashboard: {str(e)}")
        flash('เกิดข้อผิดพลาดในการโหลดข้อมูล', 'error')
        return render_template('admin_dashboard.html', form=form)

@app.route('/diagnose', methods=['POST'])
def diagnose():
    try:
        if not request.form:
            raise ValueError("ไม่พบข้อมูลที่ส่งมา")
            
        input_data = {
            'gender': request.form.get('gender'),
            'age': request.form.get('age'),
            'behavior': request.form.get('behavior'),
            'environment': request.form.get('environment'),
            'symptoms': request.form.getlist('symptoms[]')
        }

        if not all([input_data['gender'], input_data['age'], 
                   input_data['behavior'], input_data['environment']]):
            raise ValueError('กรุณากรอกข้อมูลให้ครบถ้วน')

        if not input_data['symptoms']:
            raise ValueError('กรุณาเลือกอาการอย่างน้อย 1 อาการ')

        if model is None:
            raise ValueError('ไม่พบโมเดลที่พร้อมใช้งาน')
            
        processed_data = process_input_data(input_data, label_encoders, feature_names)
        prediction_probas = model.predict_proba([processed_data])[0]
        predicted_class = model.classes_[np.argmax(prediction_probas)]
        
        top_3_indices = np.argsort(prediction_probas)[-3:][::-1]
        predictions = [
            {
                'disease': model.classes_[i],
                'probability': f"{prediction_probas[i] * 100:.1f}%",
                'info': diseases_data.get(model.classes_[i], {})
            }
            for i in top_3_indices
            if prediction_probas[i] > 0.05
        ]
        
        activity_logger.log_activity(
            'guest',
            'diagnose',
            {
                'input': input_data,
                'prediction': predicted_class,
                'probability': f"{max(prediction_probas) * 100:.1f}%"
            }
        )
        
        return render_template(
            'diagnosis.html',
            predictions=predictions,
            input_data=input_data,
            disease_info=diseases_data.get(predicted_class, {})
        )
                             
    except Exception as e:
        logger.error(f"Error in diagnosis: {str(e)}")
        flash(str(e), 'error')
        return redirect(url_for('home'))

# Error Handlers
@app.errorhandler(404)
def page_not_found(e):
    return render_template('404.html'), 404

@app.errorhandler(500)
def internal_server_error(e):
    return render_template('500.html', error=str(e)), 500

@app.errorhandler(413)
def request_entity_too_large(e):
    flash('ไฟล์มีขนาดใหญ่เกินไป (ขนาดสูงสุด 100MB)', 'error')
    return redirect(request.referrer or url_for('admin_dashboard'))

@app.errorhandler(CSRFError)
def handle_csrf_error(e):
    flash('การยืนยันความปลอดภัยล้มเหลว กรุณาลองใหม่อีกครั้ง', 'error')
    return redirect(request.referrer or url_for('home'))

if __name__ == '__main__':
    # Initialize required files and folders
    os.makedirs(Config.MODEL_FOLDER, exist_ok=True)
    os.makedirs(Config.BACKUP_FOLDER, exist_ok=True)
    os.makedirs(Config.LOG_FOLDER, exist_ok=True)
    os.makedirs('static', exist_ok=True)
    
    if not os.path.exists('diseases_data.json'):
        with open('diseases_data.json', 'w', encoding='utf-8') as f:
            json.dump({}, f, ensure_ascii=False, indent=4)
            
    # Load initial data
    model, label_encoders, feature_names = load_model()
    diseases_data = load_diseases_data()
    
    # Setup logging
    logger.info("Starting application...")
    logger.info(f"Environment: {app.config['ENV']}")
    logger.info(f"Debug mode: {app.debug}")
    logger.info("Model and disease data loaded successfully")
    
    app.run(
        host=os.getenv('FLASK_HOST', '0.0.0.0'),
        port=int(os.getenv('FLASK_PORT', 5000)),
        debug=app.debug
    )