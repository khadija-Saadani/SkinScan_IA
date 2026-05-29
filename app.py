from flask import Flask, render_template, request, redirect, url_for, session, flash, jsonify
import mysql.connector
import numpy as np
import os
from werkzeug.utils import secure_filename
from datetime import datetime
from dotenv import load_dotenv
load_dotenv()

app = Flask(__name__)
app.secret_key = os.getenv('FLASK_SECRET_KEY', 'default_secret_key_change_me')

UPLOAD_FOLDER = 'static/uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# ─── Model ────────────────────────────────────────────────────────────────────
model = None
def load_model():
    global model
    try:
        import tensorflow as tf
        model = tf.keras.models.load_model('model/vgg16_malignant_vs_benign.h5')
        print("✅ VGG16 loaded.")
    except Exception as e:
        print(f"⚠️  Demo mode: {e}")
load_model()

# ─── DB ───────────────────────────────────────────────────────────────────────
def get_db():
    return mysql.connector.connect(
        host=os.getenv('DB_HOST', 'localhost'),
        user=os.getenv('DB_USER', 'root'),
        password=os.getenv('DB_PASSWORD', ''),
        database=os.getenv('DB_NAME', 'skin_cancer_db')
    )

def allowed_file(f):
    return '.' in f and f.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# ─── Skin Validator ───────────────────────────────────────────────────────────
def is_skin_image(img_path, threshold=0.15):
    try:
        import cv2
        img = cv2.imread(img_path)
        if img is None:
            return False
        img = cv2.resize(img, (224, 224))
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        lower = np.array([0,  20,  70],  dtype=np.uint8)
        upper = np.array([25, 255, 255], dtype=np.uint8)
        mask  = cv2.inRange(hsv, lower, upper)
        skin_ratio = np.sum(mask > 0) / (224 * 224)
        return skin_ratio >= threshold
    except Exception as e:
        print(f"Skin check error: {e}")
        return True

# ─── Predict ──────────────────────────────────────────────────────────────────
MALIGNANT_THRESHOLD = 0.7  # ← tune this if needed (0.5–0.85)

def predict_image(img_path):
    if model is None:
        import hashlib
        h = int(hashlib.md5(img_path.encode()).hexdigest(), 16)
        label = 'Malignant' if h % 2 == 0 else 'Benign'
        prob  = round(60 + (h % 370) / 10, 2)
        return label, prob
    from PIL import Image
    img = Image.open(img_path).convert('RGB').resize((224, 224))
    arr = np.expand_dims(np.array(img) / 255.0, axis=0)
    score = float(model.predict(arr)[0][0])
    print(f"🔍 RAW MODEL SCORE: {score:.4f}")
    label = 'Malignant' if score >= MALIGNANT_THRESHOLD else 'Benign'
    prob  = round((score if score >= MALIGNANT_THRESHOLD else 1 - score) * 100, 2)
    print(f"🏷️  LABEL: {label} | CONFIDENCE: {prob}%")
    return label, prob

GROQ_API_KEY = os.getenv('GROQ_API_KEY', '')
GROQ_MODEL = os.getenv('GROQ_MODEL', 'mixtral-8x7b-32768')

def ask_groq(messages_history, user_message, context=None):
    from groq import Groq
    system_prompt = """You are Dr. Aiden, an expert AI medical assistant specializing in dermatology and skin cancer detection.
You assist doctors using the SkinScan AI platform — a VGG16-based binary classifier (Benign/Malignant).
Be concise, clinical, and helpful. Respond in the same language the doctor uses (French or English).
Always note AI results are screening aids, not final diagnoses."""
    if context:
        system_prompt += f"\n\nCurrent scan context: {context}"
    msgs = [{"role": "system", "content": system_prompt}]
    for m in messages_history[-10:]:
        msgs.append({"role": m["role"], "content": m["message"]})
    msgs.append({"role": "user", "content": user_message})
    try:
        client = Groq(api_key=GROQ_API_KEY)
        resp = client.chat.completions.create(model=GROQ_MODEL, messages=msgs, max_tokens=1024)
        return resp.choices[0].message.content
    except Exception as e:
        return f"⚠️ Dr. Aiden unavailable: {str(e)}"

# ─── Decorators ───────────────────────────────────────────────────────────────
from functools import wraps

def doctor_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        if 'user_id' not in session or session.get('profile') != 'doctor':
            return redirect(url_for('login'))
        return f(*args, **kwargs)
    return decorated

def patient_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        if 'patient_id' not in session or session.get('profile') != 'patient':
            return redirect(url_for('patient_login'))
        return f(*args, **kwargs)
    return decorated

# ═══════════════════════════════════════════════════════════════════════════════
#  LANDING
# ═══════════════════════════════════════════════════════════════════════════════

@app.route('/')
def landing():
    return render_template('landing.html')

# ═══════════════════════════════════════════════════════════════════════════════
#  DOCTOR AUTH
# ═══════════════════════════════════════════════════════════════════════════════

@app.route('/doctor/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username'].strip()
        password = request.form['password'].strip()
        try:
            db = get_db(); cur = db.cursor(dictionary=True)
            cur.execute("SELECT * FROM users WHERE username=%s AND password=%s", (username, password))
            user = cur.fetchone(); db.close()
            if user:
                session['user_id']   = user['id']
                session['username']  = user['username']
                session['full_name'] = user['full_name'] or user['username']
                session['role']      = user['role']
                session['profile']   = 'doctor'
                return redirect(url_for('dashboard'))
            flash('Invalid credentials.', 'danger')
        except Exception as e:
            flash(f'DB Error: {e}', 'danger')
    return render_template('login.html')

@app.route('/doctor/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username  = request.form['username'].strip()
        password  = request.form['password'].strip()
        full_name = request.form['full_name'].strip()
        confirm   = request.form['confirm'].strip()
        if password != confirm:
            flash('Passwords do not match.', 'danger')
            return redirect(url_for('register'))
        try:
            db = get_db(); cur = db.cursor()
            cur.execute("INSERT INTO users (username,password,full_name) VALUES (%s,%s,%s)",
                        (username, password, full_name))
            db.commit(); db.close()
            flash('Account created! Please log in.', 'success')
            return redirect(url_for('login'))
        except mysql.connector.IntegrityError:
            flash('Username already taken.', 'danger')
    return render_template('register.html')

@app.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('landing'))

# ═══════════════════════════════════════════════════════════════════════════════
#  PATIENT AUTH
# ═══════════════════════════════════════════════════════════════════════════════

@app.route('/patient/login', methods=['GET', 'POST'])
def patient_login():
    if request.method == 'POST':
        email    = request.form['email'].strip()
        password = request.form['password'].strip()
        try:
            db = get_db(); cur = db.cursor(dictionary=True)
            cur.execute("SELECT * FROM patient_users WHERE email=%s AND password=%s", (email, password))
            p = cur.fetchone(); db.close()
            if p:
                session['patient_id']   = p['id']
                session['patient_name'] = p['full_name']
                session['patient_email']= p['email']
                session['profile']      = 'patient'
                return redirect(url_for('patient_dashboard'))
            flash('Invalid email or password.', 'danger')
        except Exception as e:
            flash(f'Error: {e}', 'danger')
    return render_template('patient_login.html')

@app.route('/patient/register', methods=['GET', 'POST'])
def patient_register():
    if request.method == 'POST':
        full_name = request.form['full_name'].strip()
        email     = request.form['email'].strip()
        password  = request.form['password'].strip()
        confirm   = request.form['confirm'].strip()
        age       = request.form['age'].strip()
        gender    = request.form['gender'].strip()
        if password != confirm:
            flash('Passwords do not match.', 'danger')
            return redirect(url_for('patient_register'))
        try:
            db = get_db(); cur = db.cursor()
            cur.execute("INSERT INTO patient_users (full_name,email,password,age,gender) VALUES (%s,%s,%s,%s,%s)",
                        (full_name, email, password, int(age), gender))
            db.commit(); db.close()
            flash('Account created! Please log in.', 'success')
            return redirect(url_for('patient_login'))
        except mysql.connector.IntegrityError:
            flash('Email already registered.', 'danger')
        except Exception as e:
            flash(f'Error: {e}', 'danger')
    return render_template('patient_register.html')

@app.route('/patient/logout')
def patient_logout():
    session.clear()
    return redirect(url_for('landing'))

# ═══════════════════════════════════════════════════════════════════════════════
#  PATIENT PAGES
# ═══════════════════════════════════════════════════════════════════════════════

@app.route('/patient/dashboard')
@patient_required
def patient_dashboard():
    scans = []
    try:
        db = get_db(); cur = db.cursor(dictionary=True)
        cur.execute("""
            SELECT s.*, dr.message as doctor_message, dr.urgency, dr.created_at as response_date,
                   u.full_name as doctor_name
            FROM scan_requests s
            LEFT JOIN doctor_responses dr ON dr.scan_id = s.id
            LEFT JOIN users u ON u.id = dr.doctor_id
            WHERE s.patient_id = %s
            ORDER BY s.created_at DESC
        """, (session['patient_id'],))
        scans = cur.fetchall(); db.close()
    except Exception as e:
        flash(f'Error: {e}', 'danger')
    return render_template('patient_dashboard.html', scans=scans)

@app.route('/patient/scan/delete/<int:scan_id>', methods=['POST'])
@patient_required
def patient_delete_scan(scan_id):
    try:
        db = get_db(); cur = db.cursor(dictionary=True)
        cur.execute("SELECT image_path FROM scan_requests WHERE id=%s AND patient_id=%s",
                    (scan_id, session['patient_id']))
        scan = cur.fetchone()
        if scan:
            if scan['image_path']:
                img_full = os.path.join('static', scan['image_path'])
                if os.path.exists(img_full):
                    os.remove(img_full)
            cur.execute("DELETE FROM doctor_responses WHERE scan_id=%s", (scan_id,))
            cur.execute("DELETE FROM scan_requests WHERE id=%s AND patient_id=%s",
                        (scan_id, session['patient_id']))
            db.commit()
            flash('Scan deleted successfully.', 'success')
        else:
            flash('Scan not found.', 'danger')
        db.close()
    except Exception as e:
        flash(f'Error: {e}', 'danger')
    return redirect(url_for('patient_dashboard'))

@app.route('/patient/scan', methods=['GET', 'POST'])
@patient_required
def patient_scan():
    if request.method == 'POST':
        file      = request.files.get('image')
        body_part = request.form.get('body_part','').strip()
        symptoms  = request.form.get('symptoms','').strip()

        if not file or not allowed_file(file.filename):
            flash('Please upload a valid image (JPG, PNG).', 'warning')
            return redirect(url_for('patient_scan'))

        filename = secure_filename(f"p_{session['patient_id']}_{datetime.now().strftime('%Y%m%d%H%M%S')}_{file.filename}")
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        if not is_skin_image(filepath):
            os.remove(filepath)
            flash('⚠️ No skin detected in this image. Please upload a clear close-up photo of your skin lesion.', 'warning')
            return redirect(url_for('patient_scan'))

        label, prob = predict_image(filepath)
        alert_level = 'high' if label == 'Malignant' else 'none'

        try:
            db = get_db(); cur = db.cursor()
            cur.execute("""
                INSERT INTO scan_requests (patient_id, image_path, result, probability, body_part, symptoms, alert_level)
                VALUES (%s, %s, %s, %s, %s, %s, %s)
            """, (session['patient_id'], f"uploads/{filename}", label, prob/100, body_part, symptoms, alert_level))
            db.commit(); db.close()
        except Exception as e:
            flash(f'Error saving scan: {e}', 'danger')

        # PRG pattern — prevent double insert on page refresh
        session['last_patient_result'] = {
            'result': label, 'prob': prob,
            'img': f'uploads/{filename}',
            'body_part': body_part
        }
        return redirect(url_for('patient_scan_result'))

    return render_template('patient_scan.html')

@app.route('/patient/scan/result')
@patient_required
def patient_scan_result():
    result_data = session.pop('last_patient_result', None)
    if not result_data:
        return redirect(url_for('patient_scan'))
    return render_template('patient_result.html',
                           result=result_data['result'],
                           prob=result_data['prob'],
                           img=url_for('static', filename=result_data['img']),
                           body_part=result_data['body_part'])

# ═══════════════════════════════════════════════════════════════════════════════
#  DOCTOR PAGES
# ═══════════════════════════════════════════════════════════════════════════════

@app.route('/doctor/dashboard')
@doctor_required
def dashboard():
    stats  = {'total': 0, 'malignant': 0, 'benign': 0, 'today': 0, 'alerts': 0}
    recent = []
    try:
        db = get_db(); cur = db.cursor(dictionary=True)
        cur.execute("SELECT COUNT(*) as c FROM patients WHERE user_id=%s", (session['user_id'],))
        stats['total'] = cur.fetchone()['c']
        cur.execute("SELECT COUNT(*) as c FROM patients WHERE user_id=%s AND result='Malignant'", (session['user_id'],))
        stats['malignant'] = cur.fetchone()['c']
        stats['benign'] = stats['total'] - stats['malignant']
        cur.execute("SELECT COUNT(*) as c FROM patients WHERE user_id=%s AND DATE(created_at)=CURDATE()", (session['user_id'],))
        stats['today'] = cur.fetchone()['c']
        cur.execute("SELECT COUNT(*) as c FROM scan_requests WHERE alert_level='high' AND status='pending'")
        stats['alerts'] = cur.fetchone()['c']
        cur.execute("SELECT * FROM patients WHERE user_id=%s ORDER BY created_at DESC LIMIT 5", (session['user_id'],))
        recent = cur.fetchall()
        db.close()
    except Exception as e:
        flash(f'DB Error: {e}', 'danger')
    return render_template('dashboard.html', stats=stats, recent=recent, now=datetime.now())

@app.route('/doctor/predict', methods=['GET', 'POST'])
@doctor_required
def predict():
    if request.method == 'POST':
        name  = request.form.get('name','').strip()
        age   = request.form.get('age','').strip()
        notes = request.form.get('notes','').strip()
        file  = request.files.get('image')
        if not name or not age or not file:
            flash('Please fill all fields.', 'warning')
            return redirect(url_for('predict'))
        if file and allowed_file(file.filename):
            filename = secure_filename(f"{datetime.now().strftime('%Y%m%d%H%M%S')}_{file.filename}")
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            if not is_skin_image(filepath):
                os.remove(filepath)
                flash('⚠️ No skin detected in this image. Please upload a clear photo of the skin lesion.', 'warning')
                return redirect(url_for('predict'))
            label, prob = predict_image(filepath)
            try:
                db = get_db(); cur = db.cursor()
                cur.execute("INSERT INTO patients (user_id,name,age,result,probability,image_path,notes) VALUES (%s,%s,%s,%s,%s,%s,%s)",
                            (session['user_id'], name, int(age), label, prob/100, f"uploads/{filename}", notes))
                db.commit(); db.close()
            except Exception as e:
                flash(f'DB Error: {e}', 'danger')
            # PRG pattern: store in session, redirect to prevent double-submit on refresh
            session['last_result'] = {
                'result': label, 'prob': prob,
                'img': f'uploads/{filename}',
                'name': name, 'age': age, 'notes': notes
            }
            return redirect(url_for('predict_result'))
        flash('Unsupported format.', 'danger')
    return render_template('predict.html')

@app.route('/doctor/predict/result')
@doctor_required
def predict_result():
    result_data = session.pop('last_result', None)
    if not result_data:
        return redirect(url_for('predict'))
    return render_template('result.html',
                           result=result_data['result'],
                           prob=result_data['prob'],
                           img=url_for('static', filename=result_data['img']),
                           name=result_data['name'],
                           age=result_data['age'],
                           notes=result_data['notes'])

@app.route('/doctor/patients')
@doctor_required
def patients():
    search = request.args.get('q','').strip()
    rows = []
    try:
        db = get_db(); cur = db.cursor(dictionary=True)
        if search:
            cur.execute("SELECT * FROM patients WHERE user_id=%s AND name LIKE %s ORDER BY created_at DESC",
                        (session['user_id'], f'%{search}%'))
        else:
            cur.execute("SELECT * FROM patients WHERE user_id=%s ORDER BY created_at DESC", (session['user_id'],))
        rows = cur.fetchall(); db.close()
    except Exception as e:
        flash(f'Error: {e}', 'danger')
    return render_template('patients.html', patients=rows, search=search)

@app.route('/doctor/patients/delete/<int:patient_id>', methods=['POST'])
@doctor_required
def delete_patient(patient_id):
    try:
        db = get_db(); cur = db.cursor(dictionary=True)
        # Make sure record belongs to this doctor
        cur.execute("SELECT image_path FROM patients WHERE id=%s AND user_id=%s",
                    (patient_id, session['user_id']))
        record = cur.fetchone()
        if record:
            if record['image_path']:
                img_full = os.path.join('static', record['image_path'])
                if os.path.exists(img_full):
                    os.remove(img_full)
            cur.execute("DELETE FROM patients WHERE id=%s AND user_id=%s",
                        (patient_id, session['user_id']))
            db.commit()
            flash('Record deleted successfully.', 'success')
        else:
            flash('Record not found.', 'danger')
        db.close()
    except Exception as e:
        flash(f'Error: {e}', 'danger')
    return redirect(url_for('patients'))

@app.route('/doctor/alerts')
@doctor_required
def alerts():
    scans = []
    try:
        db = get_db(); cur = db.cursor(dictionary=True)
        cur.execute("""
            SELECT s.*, p.full_name as patient_name, p.age, p.gender, p.email,
                   dr.message as response, dr.urgency, dr.created_at as response_date,
                   u.full_name as doctor_name
            FROM scan_requests s
            JOIN patient_users p ON p.id = s.patient_id
            LEFT JOIN doctor_responses dr ON dr.scan_id = s.id
            LEFT JOIN users u ON u.id = dr.doctor_id
            WHERE s.alert_level = 'high'
            ORDER BY s.status ASC, s.created_at DESC
        """)
        scans = cur.fetchall(); db.close()
    except Exception as e:
        flash(f'Error: {e}', 'danger')
    return render_template('alerts.html', scans=scans)

@app.route('/doctor/respond/<int:scan_id>', methods=['POST'])
@doctor_required
def respond(scan_id):
    message = request.form.get('message','').strip()
    urgency = request.form.get('urgency', 'normal')
    if not message:
        flash('Please write a response.', 'warning')
        return redirect(url_for('alerts'))
    try:
        db = get_db(); cur = db.cursor()
        cur.execute("DELETE FROM doctor_responses WHERE scan_id=%s", (scan_id,))
        cur.execute("INSERT INTO doctor_responses (scan_id, doctor_id, message, urgency) VALUES (%s,%s,%s,%s)",
                    (scan_id, session['user_id'], message, urgency))
        cur.execute("UPDATE scan_requests SET status='reviewed' WHERE id=%s", (scan_id,))
        db.commit(); db.close()
        flash('Response sent to patient ✅', 'success')
    except Exception as e:
        flash(f'Error: {e}', 'danger')
    return redirect(url_for('alerts'))

# ═══════════════════════════════════════════════════════════════════════════════
#  PATIENT AI — ARIA
# ═══════════════════════════════════════════════════════════════════════════════

def ask_aria(messages_history, user_message, patient_context=None):
    from groq import Groq
    system_prompt = """You are Aria, a warm and caring AI health guide for patients using SkinScan AI.

Your role is to support PATIENTS — not doctors. They may be scared, confused, or anxious after receiving a skin scan result.

Your personality:
- Warm, calm, empathetic and reassuring
- Use simple everyday language — no medical jargon
- Never be alarmist — be honest but gentle
- Encourage action without causing panic

What you help with:
- Explaining what Benign and Malignant mean in simple terms
- Helping patients understand their confidence score
- Explaining what to expect at a dermatologist visit
- Answering questions about skin cancer prevention and detection
- Explaining the ABCDE rule in simple terms
- Encouraging healthy sun protection habits
- Calming fears and anxiety about scan results

Important rules:
- ALWAYS remind the patient that the AI scan is a screening tool, not a final diagnosis
- ALWAYS encourage them to see a doctor, especially for malignant results
- NEVER diagnose or prescribe anything
- NEVER say anything that could replace medical advice
- If a patient seems very distressed, acknowledge their feelings first before giving information
- Keep responses concise and easy to read — use short paragraphs or bullet points
- Respond in the same language the patient uses (English or French)
"""
    if patient_context:
        system_prompt += f"\n\nThis patient's latest scan result: {patient_context}"

    msgs = [{"role": "system", "content": system_prompt}]
    for m in messages_history[-10:]:
        msgs.append({"role": m["role"], "content": m["message"]})
    msgs.append({"role": "user", "content": user_message})

    try:
        client = Groq(api_key=GROQ_API_KEY)
        resp = client.chat.completions.create(model=GROQ_MODEL, messages=msgs, max_tokens=800)
        return resp.choices[0].message.content
    except Exception as e:
        return f"I'm having a little trouble right now. Please try again in a moment. ({str(e)})"

@app.route('/patient/ai')
@patient_required
def patient_ai():
    history = []
    latest_scan = None
    try:
        db = get_db(); cur = db.cursor(dictionary=True)
        cur.execute("SELECT role, message FROM chat_history WHERE user_id=%s ORDER BY created_at ASC LIMIT 40",
                    (session['patient_id'],))
        history = cur.fetchall()
        cur.execute("SELECT * FROM scan_requests WHERE patient_id=%s ORDER BY created_at DESC LIMIT 1",
                    (session['patient_id'],))
        latest_scan = cur.fetchone()
        db.close()
    except: pass
    return render_template('patient_ai.html', history=history, latest_scan=latest_scan)

@app.route('/patient/ai/chat', methods=['POST'])
@patient_required
def patient_ai_chat():
    data     = request.get_json()
    user_msg = data.get('message','').strip()
    if not user_msg:
        return jsonify({'error': 'Empty'}), 400

    history = []
    patient_context = None
    db = None
    try:
        db = get_db(); cur = db.cursor(dictionary=True)
        cur.execute("SELECT role, message FROM chat_history WHERE user_id=%s ORDER BY created_at ASC LIMIT 20",
                    (session['patient_id'],))
        history = cur.fetchall()
        cur.execute("SELECT result, probability, body_part FROM scan_requests WHERE patient_id=%s ORDER BY created_at DESC LIMIT 1",
                    (session['patient_id'],))
        scan = cur.fetchone()
        if scan:
            patient_context = f"{scan['result']} at {round(scan['probability']*100,1)}% confidence on {scan['body_part'] or 'skin'}"
    except: pass

    reply = ask_aria(history, user_msg, patient_context)

    try:
        cur.execute("INSERT INTO chat_history (user_id,role,message) VALUES (%s,'user',%s)",
                    (session['patient_id'], user_msg))
        cur.execute("INSERT INTO chat_history (user_id,role,message) VALUES (%s,'assistant',%s)",
                    (session['patient_id'], reply))
        db.commit(); db.close()
    except: pass

    return jsonify({'reply': reply})

@app.route('/patient/ai/clear', methods=['POST'])
@patient_required
def patient_ai_clear():
    try:
        db = get_db(); cur = db.cursor()
        cur.execute("DELETE FROM chat_history WHERE user_id=%s", (session['patient_id'],))
        db.commit(); db.close()
    except: pass
    return jsonify({'ok': True})

@app.route('/doctor/alerts/count')
@doctor_required
def alerts_count():
    try:
        db = get_db(); cur = db.cursor(dictionary=True)
        cur.execute("SELECT COUNT(*) as c FROM scan_requests WHERE alert_level='high' AND status='pending'")
        count = cur.fetchone()['c']; db.close()
        return jsonify({'count': count})
    except:
        return jsonify({'count': 0})

if __name__ == '__main__':
    app.run(debug=True)