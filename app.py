from flask import Flask, render_template, request, redirect, url_for, flash, jsonify, session
from flask_wtf.csrf import CSRFProtect
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
from functools import wraps
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
import traceback

# ตั้งค่า logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('app.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# สร้าง Flask app
app = Flask(__name__)

app.config.update(
    SECRET_KEY=os.getenv('SECRET_KEY'),
    SESSION_TYPE='filesystem',
    PERMANENT_SESSION_LIFETIME=timedelta(minutes=int(os.getenv('SESSION_LIFETIME', 30))),
    MAX_CONTENT_LENGTH=16 * 1024 * 1024,  # 16MB max file size
    UPLOAD_FOLDER=os.getenv('UPLOAD_FOLDER', 'uploads'),
    ENV=os.getenv('FLASK_ENV', 'development'),
    DEBUG=os.getenv('FLASK_ENV', 'development') == 'development',
    WTF_CSRF_SECRET_KEY=os.getenv('WTF_CSRF_SECRET_KEY'),
    WTF_CSRF_TIME_LIMIT=None,
    SQLALCHEMY_DATABASE_URI=os.getenv('DATABASE_URL', 'sqlite:///your_database.db'),
    SQLALCHEMY_TRACK_MODIFICATIONS=False
)

# ตั้งค่า CSRF protection
csrf = CSRFProtect()
csrf.init_app(app)

# ตั้งค่า Rate Limiting
limiter = Limiter(
    app=app,
    key_func=get_remote_address,
    default_limits=["200 per day", "50 per hour"]
)

# Constants
ADMIN_USERNAME = os.getenv('ADMIN_USERNAME', 'admin')
ADMIN_PASSWORD_HASH = os.getenv('ADMIN_PASSWORD_HASH', generate_password_hash('admin'))
MODEL_FOLDER = 'models'
BACKUP_FOLDER = 'backups'
ALLOWED_EXTENSIONS = {'joblib', 'pkl'}
MODEL_FILENAME = 'horse_disease_model.joblib'
MODEL_PATH = os.path.join(MODEL_FOLDER, MODEL_FILENAME)

# สร้างโฟลเดอร์ที่จำเป็น
for folder in [MODEL_FOLDER, 'static', BACKUP_FOLDER, app.config['UPLOAD_FOLDER'], 'logs']:
    os.makedirs(folder, exist_ok=True)

# Custom Exceptions
class CustomError(Exception):
    def __init__(self, message, status_code=400):
        self.message = message
        self.status_code = status_code

# Activity Logger
class ActivityLog:
    def __init__(self, user, action, details=None):
        self.timestamp = datetime.now()
        self.user = user
        self.action = action
        self.details = details

    def save(self):
        try:
            log_entry = {
                'timestamp': self.timestamp.isoformat(),
                'user': self.user,
                'action': self.action,
                'details': self.details,
                'ip': request.remote_addr
            }
            log_file = os.path.join('logs', f'activity_{self.timestamp.strftime("%Y%m")}.json')
            
            existing_logs = []
            if os.path.exists(log_file):
                with open(log_file, 'r', encoding='utf-8') as f:
                    existing_logs = json.load(f)
            
            existing_logs.append(log_entry)
            
            with open(log_file, 'w', encoding='utf-8') as f:
                json.dump(existing_logs, f, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.error(f"Failed to save activity log: {str(e)}")

# Utility Functions
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def handle_error(e, message="เกิดข้อผิดพลาด", log_level=logging.ERROR):
    logger.log(log_level, f"{message}: {str(e)}", exc_info=True)
    flash(message, 'error')

def validate_model_file(file):
    if not file:
        raise CustomError('ไม่พบไฟล์ที่อัปโหลด')
    if not allowed_file(file.filename):
        raise CustomError('ประเภทไฟล์ไม่ถูกต้อง')
    return True

def load_diseases_data():
    try:
        if not os.path.exists('diseases_data.json'):
            with open('diseases_data.json', 'w', encoding='utf-8') as f:
                json.dump({}, f, ensure_ascii=False, indent=4)
        with open('diseases_data.json', 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        handle_error(e, "ไม่สามารถโหลดข้อมูลโรคได้")
        return {}

def save_diseases_data():
    try:
        temp_file = 'diseases_data.json.tmp'
        with open(temp_file, 'w', encoding='utf-8') as f:
            json.dump(diseases_data, f, ensure_ascii=False, indent=4)
        os.replace(temp_file, 'diseases_data.json')
        return True
    except Exception as e:
        if os.path.exists(temp_file):
            os.remove(temp_file)
        handle_error(e, "ไม่สามารถบันทึกข้อมูลโรคได้")
        return False

def is_valid_model_file(file_path):
    try:
        test_model = joblib.load(file_path)
        required_keys = ['model', 'label_encoders', 'feature_names']
        return all(key in test_model for key in required_keys)
    except Exception:
        return False

def load_model(model_path=MODEL_PATH):
    if os.path.exists(model_path):
        try:
            model_data = joblib.load(model_path)
            return model_data['model'], model_data['label_encoders'], model_data['feature_names']
        except Exception as e:
            logger.error(f"เกิดข้อผิดพลาดในการโหลดโมเดล: {str(e)}", exc_info=True)
    return None, None, None

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
        logger.error(f"Error processing input data: {str(e)}", exc_info=True)
        raise CustomError("เกิดข้อผิดพลาดในการประมวลผลข้อมูล")

def cleanup_old_files(folder, max_age_days=30):
    try:
        cutoff = datetime.now() - timedelta(days=max_age_days)
        for filename in os.listdir(folder):
            filepath = os.path.join(folder, filename)
            if os.path.getctime(filepath) < cutoff.timestamp():
                if os.path.isfile(filepath):
                    os.remove(filepath)
                elif os.path.isdir(filepath):
                    shutil.rmtree(filepath)
    except Exception as e:
        logger.error(f"Error cleaning up old files: {str(e)}")

def backup_system():
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    backup_dir = os.path.join(BACKUP_FOLDER, f'backup_{timestamp}')
    
    try:
        os.makedirs(backup_dir, exist_ok=True)
        
        if os.path.exists(MODEL_FOLDER):
            shutil.copytree(MODEL_FOLDER, os.path.join(backup_dir, 'models'))
        
        if os.path.exists('diseases_data.json'):
            shutil.copy2('diseases_data.json', backup_dir)
            
        if os.path.exists('logs'):
            shutil.copytree('logs', os.path.join(backup_dir, 'logs'))
            
        backup_info = {
            'timestamp': timestamp,
            'models_count': len(os.listdir(MODEL_FOLDER)) if os.path.exists(MODEL_FOLDER) else 0,
            'diseases_count': len(diseases_data),
            'backup_size': sum(os.path.getsize(os.path.join(dirpath, filename))
                             for dirpath, _, filenames in os.walk(backup_dir)
                             for filename in filenames)
        }
        
        with open(os.path.join(backup_dir, 'backup_info.json'), 'w', encoding='utf-8') as f:
            json.dump(backup_info, f, ensure_ascii=False, indent=2)
            
        return True, backup_dir
    except Exception as e:
        logger.error(f"Backup failed: {str(e)}")
        return False, str(e)

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

# Security Middleware
@app.before_request
def before_request():
    if not request.is_secure and app.env != 'development':
        url = request.url.replace('http://', 'https://', 1)
        return redirect(url, code=301)

@app.after_request
def after_request(response):
    response.headers['X-Content-Type-Options'] = 'nosniff'
    response.headers['X-Frame-Options'] = 'SAMEORIGIN'
    response.headers['X-XSS-Protection'] = '1; mode=block'
    return response

# Routes
@app.route('/')
def home():
    try:
        return render_template('index.html')
    except Exception as e:
        logger.error(f"Error in home route: {str(e)}", exc_info=True)
        return render_template('500.html', error=str(e)), 500

@app.route('/about')
def about():
    try:
        return render_template('about.html')
    except Exception as e:
        logger.error(f"Error in about route: {str(e)}", exc_info=True)
        return render_template('500.html', error=str(e)), 500

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
                
                ActivityLog(
                    user=username,
                    action='login',
                    details={'ip': request.remote_addr}
                ).save()
                
                logger.info(f"ผู้ใช้ {username} เข้าสู่ระบบสำเร็จ")
                flash('เข้าสู่ระบบสำเร็จ', 'success')
                return redirect(url_for('admin_dashboard'))
            else:
                logger.warning(f'การเข้าสู่ระบบล้มเหลว สำหรับผู้ใช้: {username}')
                flash('ชื่อผู้ใช้หรือรหัสผ่านไม่ถูกต้อง', 'error')
                
        except Exception as e:
            logger.error(f"Error in login: {str(e)}", exc_info=True)
            flash('เกิดข้อผิดพลาดในการเข้าสู่ระบบ', 'error')
    
    return render_template('admin_login.html')

@app.route('/admin/logout', methods=['GET', 'POST'])
def admin_logout():
    if session.get('username'):
        ActivityLog(
            user=session['username'],
            action='logout'
        ).save()
    
    session.clear()
    flash('คุณได้ออกจากระบบแล้ว', 'success')
    return redirect(url_for('home'))

@app.route('/admin/dashboard', methods=['GET', 'POST'])
@login_required
def admin_dashboard():
    try:
        if request.method == 'POST':
            action = request.form.get('action')
            
            try:
                if action == 'add_model':
                    if 'model_file' not in request.files:
                        raise CustomError('ไม่พบไฟล์ที่อัปโหลด')
                    
                    file = request.files['model_file']
                    validate_model_file(file)
                    
                    filename = secure_filename(file.filename)
                    filepath = os.path.join(MODEL_FOLDER, filename)
                    
                    temp_path = os.path.join(MODEL_FOLDER, f'temp_{filename}')
                    file.save(temp_path)
                    
                    if not is_valid_model_file(temp_path):
                        os.remove(temp_path)
                        raise CustomError('ไฟล์โมเดลไม่ถูกต้อง')
                    
                    os.replace(temp_path, filepath)
                    
                    ActivityLog(
                        user=session['username'],
                        action='add_model',
                        details={'model_name': filename}
                    ).save()
                    
                    flash('อัปโหลดโมเดลสำเร็จ', 'success')
                    
                elif action == 'delete_model':
                    model_name = request.form.get('model_name')
                    filepath = os.path.join(MODEL_FOLDER, model_name)
                    
                    if os.path.exists(filepath):
                        backup_path = os.path.join(BACKUP_FOLDER, f'deleted_{model_name}_{datetime.now().strftime("%Y%m%d_%H%M%S")}')
                        shutil.copy2(filepath, backup_path)
                        
                        os.remove(filepath)
                        ActivityLog(
                            user=session['username'],
                            action='delete_model',
                            details={'model_name': model_name}
                        ).save()
                        flash('ลบโมเดลสำเร็จ', 'success')
                    else:
                        raise CustomError('ไม่พบไฟล์โมเดล')
                        
                elif action == 'set_active':
                    model_name = request.form.get('model_name')
                    new_path = os.path.join(MODEL_FOLDER, model_name)
                    
                    if not os.path.exists(new_path):
                        raise CustomError('ไม่พบไฟล์โมเดล')
                    
                    global model, label_encoders, feature_names
                    model, label_encoders, feature_names = load_model(new_path)
                    
                    if model is None:
                        raise CustomError('ไม่สามารถโหลดโมเดลได้')
                    
                    ActivityLog(
                        user=session['username'],
                        action='set_active_model',
                        details={'model_name': model_name}
                    ).save()
                    
                    flash('เปลี่ยนโมเดลที่ใช้งานสำเร็จ', 'success')
                    
                elif action in ['add_disease', 'edit_disease', 'delete_disease']:
                    global diseases_data
                    
                    if action == 'add_disease':
                        disease_name = request.form.get('disease_name')
                        if disease_name in diseases_data:
                            raise CustomError('มีข้อมูลโรคนี้อยู่แล้ว')
                        
                        diseases_data[disease_name] = {
                            'ชื่อโรค (ไทย)': disease_name,
                            'ชื่อโรค (อังกฤษ)': request.form.get('disease_name_en'),
                            'อาการ': request.form.get('symptoms'),
                            'การควบคุมและป้องกัน': request.form.get('prevention'),
                            'ระดับความรุนแรง': request.form.get('severity'),
                            'last_updated': datetime.now().isoformat(),
                            'updated_by': session['username']
                        }
                        
                        ActivityLog(
                            user=session['username'],
                            action='add_disease',
                            details={'disease_name': disease_name}
                        ).save()
                        
                    elif action == 'edit_disease':
                        index = int(request.form.get('index'))
                        disease_keys = list(diseases_data.keys())
                        if index >= len(disease_keys):
                            raise CustomError('ไม่พบข้อมูลโรค')
                        
                        disease_name = disease_keys[index]
                        diseases_data[disease_name].update({
                            'ชื่อโรค (ไทย)': request.form.get('disease_name'),
                            'ชื่อโรค (อังกฤษ)': request.form.get('disease_name_en'),
                            'อาการ': request.form.get('symptoms'),
                            'การควบคุมและป้องกัน': request.form.get('prevention'),
                            'ระดับความรุนแรง': request.form.get('severity'),
                            'last_updated': datetime.now().isoformat(),
                            'updated_by': session['username']
                        })
                        
                        ActivityLog(
                            user=session['username'],
                            action='edit_disease',
                            details={'disease_name': disease_name}
                        ).save()
                        
                    elif action == 'delete_disease':
                        index = int(request.form.get('index'))
                        disease_keys = list(diseases_data.keys())
                        if index >= len(disease_keys):
                            raise CustomError('ไม่พบข้อมูลโรค')
                        
                        disease_name = disease_keys[index]
                        backup = {
                            'deleted_at': datetime.now().isoformat(),
                            'deleted_by': session['username'],
                            'disease_data': diseases_data[disease_name]
                        }
                        
                        deleted_file = os.path.join(BACKUP_FOLDER, f'deleted_disease_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json')
                        with open(deleted_file, 'w', encoding='utf-8') as f:
                            json.dump(backup, f, ensure_ascii=False, indent=4)
                        
                        del diseases_data[disease_name]
                        
                        ActivityLog(
                            user=session['username'],
                            action='delete_disease',
                            details={'disease_name': disease_name}
                        ).save()
                    
                    if not save_diseases_data():
                        raise CustomError('ไม่สามารถบันทึกข้อมูลได้')
                    
                    flash('ดำเนินการเกี่ยวกับข้อมูลโรคสำเร็จ', 'success')
                    
                elif action == 'backup':
                    success, result = backup_system()
                    if success:
                        ActivityLog(
                            user=session['username'],
                            action='create_backup',
                            details={'backup_path': result}
                        ).save()
                        flash(f'สำรองข้อมูลสำเร็จที่ {result}', 'success')
                    else:
                        raise CustomError(f'เกิดข้อผิดพลาดในการสำรองข้อมูล: {result}')
                        
            except CustomError as e:
                flash(e.message, 'error')
            except Exception as e:
                logger.error(f"Error in dashboard action {action}: {str(e)}", exc_info=True)
                flash('เกิดข้อผิดพลาดที่ไม่คาดคิด', 'error')
            
            return redirect(url_for('admin_dashboard'))
        
        # GET request
        cleanup_old_files(BACKUP_FOLDER, max_age_days=30)
        
        models = []
        if os.path.exists(MODEL_FOLDER):
            for filename in os.listdir(MODEL_FOLDER):
                if filename.endswith(('.joblib', '.pkl')):
                    file_path = os.path.join(MODEL_FOLDER, filename)
                    models.append({
                        'model_name': filename,
                        'upload_date': datetime.fromtimestamp(
                            os.path.getctime(file_path)
                        ).strftime('%Y-%m-%d %H:%M:%S'),
                        'status': 'active' if filename == os.path.basename(MODEL_PATH) else 'inactive',
                        'size': f"{os.path.getsize(file_path) / (1024*1024):.2f} MB"
                    })
        
        diseases = [
            {**disease_data, 'key': disease_name} 
            for disease_name, disease_data in diseases_data.items()
        ]
        
        stats = {
            'total_models': len(models),
            'active_models': len([m for m in models if m['status'] == 'active']),
            'total_diseases': len(diseases),
            'disk_usage': {
                'models': sum(os.path.getsize(os.path.join(MODEL_FOLDER, f)) 
                            for f in os.listdir(MODEL_FOLDER)) / (1024*1024) if os.path.exists(MODEL_FOLDER) else 0,
                'backups': sum(os.path.getsize(os.path.join(BACKUP_FOLDER, f)) 
                             for f in os.listdir(BACKUP_FOLDER)) / (1024*1024) if os.path.exists(BACKUP_FOLDER) else 0
            }
        }
        
        return render_template('admin_dashboard.html',
                             models=models,
                             diseases=diseases,
                             stats=stats,
                             current_model=os.path.basename(MODEL_PATH) if model else None)
                             
    except Exception as e:
        logger.error("Error loading dashboard data", exc_info=True)
        flash('เกิดข้อผิดพลาดในการโหลดข้อมูล', 'error')
        return render_template('admin_dashboard.html')

@app.route('/diagnose', methods=['POST'])
def diagnose():
    try:
        input_data = {
            'gender': request.form.get('gender'),
            'age': request.form.get('age'),
            'behavior': request.form.get('behavior'),
            'environment': request.form.get('environment'),
            'symptoms': request.form.getlist('symptoms[]')
        }

        if not all(input_data.values()):
            raise CustomError('กรุณากรอกข้อมูลให้ครบถ้วน')

        if model is None:
            raise CustomError('ไม่พบโมเดลที่พร้อมใช้งาน')
            
        processed_data = process_input_data(input_data, label_encoders, feature_names)
        
        try:
            prediction_probas = model.predict_proba([processed_data])[0]
            predicted_class = model.predict([processed_data])[0]
            
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
            
            return render_template(
                'diagnosis.html',
                predictions=predictions,
                input_data=input_data
            )
                             
        except Exception as e:
            logger.error(f"Prediction error: {str(e)}", exc_info=True)
            raise CustomError('เกิดข้อผิดพลาดในการวินิจฉัย')
            
    except CustomError as e:
        flash(e.message, 'error')
        return redirect(url_for('home'))
    except Exception as e:
        logger.error("Error in diagnose route", exc_info=True)
        flash('เกิดข้อผิดพลาดในการวินิจฉัย', 'error')
        return render_template('500.html', error=str(e)), 500

# Error Handlers
@app.errorhandler(404)
def page_not_found(e):
    return render_template('404.html'), 404

@app.errorhandler(500)
def internal_server_error(e):
    handle_error(e, "เกิดข้อผิดพลาดภายในเซิร์ฟเวอร์")
    return render_template('500.html'), 500

@app.errorhandler(CustomError)
def handle_custom_error(error):
    flash(error.message, 'error')
    return redirect(request.referrer or url_for('home'))

@app.errorhandler(Exception)
def handle_all_exceptions(e):
    logger.error("Unhandled Exception", exc_info=True)
    return render_template('500.html', error=str(e)), 500

if __name__ == '__main__':
    # Initialize required files
    if not os.path.exists('diseases_data.json'):
        with open('diseases_data.json', 'w', encoding='utf-8') as f:
            json.dump({}, f, ensure_ascii=False, indent=4)
            
    # Load initial data
    model, label_encoders, feature_names = load_model()
    diseases_data = load_diseases_data()
    
    logger.info("Starting application...")
    logger.info(f"Environment: {app.config['ENV']}")
    logger.info(f"Debug mode: {app.debug}")
    logger.info("Model and disease data loaded successfully")
    
    app.run(
        host=os.getenv('FLASK_HOST', '0.0.0.0'),
        port=int(os.getenv('FLASK_PORT', 5000)),
        debug=app.debug,
        ssl_context='adhoc' if app.env != 'development' else None
    )