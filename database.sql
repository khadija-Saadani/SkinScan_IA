
CREATE DATABASE IF NOT EXISTS skin_cancer_db;
USE skin_cancer_db;

CREATE TABLE IF NOT EXISTS users (
    id         INT AUTO_INCREMENT PRIMARY KEY,
    username   VARCHAR(50) UNIQUE NOT NULL,
    password   VARCHAR(255) NOT NULL,
    full_name  VARCHAR(100),
    role       VARCHAR(20) DEFAULT 'doctor',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS patient_users (
    id         INT AUTO_INCREMENT PRIMARY KEY,
    full_name  VARCHAR(100) NOT NULL,
    email      VARCHAR(100) UNIQUE NOT NULL,
    password   VARCHAR(255) NOT NULL,
    age        INT,
    gender     VARCHAR(10),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
CREATE TABLE IF NOT EXISTS scan_requests (
    id           INT AUTO_INCREMENT PRIMARY KEY,
    patient_id   INT NOT NULL,
    image_path   VARCHAR(255),
    result       VARCHAR(20),
    probability  FLOAT,
    body_part    VARCHAR(100),
    symptoms     TEXT,
    status       VARCHAR(20) DEFAULT 'pending',  
    alert_level  VARCHAR(10) DEFAULT 'low',      
    created_at   TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (patient_id) REFERENCES patient_users(id)
);

CREATE TABLE IF NOT EXISTS doctor_responses (
    id            INT AUTO_INCREMENT PRIMARY KEY,
    scan_id       INT NOT NULL,
    doctor_id     INT NOT NULL,
    message       TEXT NOT NULL,
    urgency       VARCHAR(20) DEFAULT 'normal', 
    created_at    TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (scan_id)   REFERENCES scan_requests(id),
    FOREIGN KEY (doctor_id) REFERENCES users(id)
);

CREATE TABLE IF NOT EXISTS chat_history (
    id         INT AUTO_INCREMENT PRIMARY KEY,
    user_id    INT,
    role       VARCHAR(10),
    message    TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (user_id) REFERENCES users(id)
);

CREATE TABLE IF NOT EXISTS patients (
    id          INT AUTO_INCREMENT PRIMARY KEY,
    user_id     INT,
    name        VARCHAR(100),
    age         INT,
    result      VARCHAR(20),
    probability FLOAT,
    image_path  VARCHAR(255),
    notes       TEXT,
    created_at  TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (user_id) REFERENCES users(id)
);

INSERT IGNORE INTO users (username, password, full_name, role)
VALUES ('admin', '1234', 'Dr. Admin', 'admin');