import sqlite3
from flask import Flask, jsonify, request
import os 
import threading
import time
from helper_files.reconcile import reconcile

app = Flask(__name__)

DB_PATH = "data/ocr_results.db"
os.makedirs("data", exist_ok=True)

def init_db():
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    cursor = conn.cursor()

    # Raw OCR table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS ocr_results (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            frame_id INTEGER,
            timestamp REAL,
            vehicle_track_id INTEGER,
            rec_confidence REAL,
            rec_text TEXT
        )
    """)

    # Reconciled results table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS vehicle_plates (
            vehicle_track_id INTEGER PRIMARY KEY,
            final_plate TEXT,
            vote_count INTEGER,
            avg_confidence REAL,
            resolved_at REAL
        )
    """)

    cursor.execute("""CREATE UNIQUE INDEX IF NOT EXISTS idx_vehicle_unique ON vehicle_plates(vehicle_track_id);""")
    
    conn.commit()
    return conn


db_conn = init_db()
db_cursor = db_conn.cursor()
print("DB Initialized")

def reconcile_loop():
    while True:
        try:
            reconcile()
        except Exception as e:
            print(f"[RECONCILE ERROR] {e}")
        time.sleep(1)  # every 1 second

threading.Thread(target=reconcile_loop, daemon=True).start()

@app.route("/ingest", methods = ["POST"])
def insert():
    data = request.json 
    frame_id = data["frame_id"]
    timestamp = data["timestamp"]
    vehicle_track_id = data["vehicle_track_id"]
    rec_confidence = data["rec_confidence"]
    rec_text = data["rec_text"]
    db_cursor.execute("""INSERT INTO ocr_results (frame_id, timestamp, vehicle_track_id, rec_confidence, rec_text) VALUES (?, ?, ?, ?, ?)""",
                      (frame_id, timestamp, vehicle_track_id, float(rec_confidence),rec_text))
    db_conn.commit()
    return jsonify({"status": "ok"}), 200

@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ingestion up"})

if __name__ == "__main__":
    app.run(host = "0.0.0.0", port=8004, debug=False)
