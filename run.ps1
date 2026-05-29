# Activate virtual environment and run Flask app
.\venv\Scripts\activate
$env:FLASK_APP = "app.py"
$env:FLASK_ENV = "development"
python app.py
