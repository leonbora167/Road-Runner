import streamlit as st
import subprocess
import tempfile
import time
import os
import signal
import sqlite3
import pandas as pd

st.set_page_config(layout="wide")
st.title("Road-Runner")

# ---------------------------
# Session State
# ---------------------------
if "infra_procs" not in st.session_state:
    st.session_state.infra_procs = []

if "ai_procs" not in st.session_state:
    st.session_state.ai_procs = []

if "stream_proc" not in st.session_state:
    st.session_state.stream_proc = None

if "video_path" not in st.session_state:
    st.session_state.video_path = None


# ---------------------------
# Command Definitions
# ---------------------------

INFRA_COMMANDS = [
    ["python", "-m", "helper_files.data_utilities"],  # OCR ingestion service
]

AI_SERVICES = [
    ["python", "-m", "ai_services.yolo_vehicle"],
    ["python", "-m", "ai_services.vehicle_tracker"],
    ["python", "-m", "ai_services.yolo_number_plate"],
    ["python", "-m", "ai_services.ocr1"],
]


# ---------------------------
# Helpers
# ---------------------------

def start_processes(commands, target_list):
    for cmd in commands:
        p = subprocess.Popen(cmd)
        target_list.append(p)


def stop_processes(processes):
    for p in processes:
        try:
            p.terminate()
        except Exception:
            pass
    processes.clear()


def stop_process(proc):
    if proc:
        try:
            proc.terminate()
        except Exception:
            pass

def fetch_reconciled_results():
    try:
        conn = sqlite3.connect("data/ocr_results.db")
        df = pd.read_sql_query("""
            SELECT
                vehicle_track_id AS "Vehicle ID",
                final_plate AS "Plate",
                ROUND(avg_confidence, 2) AS "Confidence"
            FROM vehicle_plates
            ORDER BY resolved_at DESC
            LIMIT 10
        """, conn)
        conn.close()
        return df
    except Exception:
        # Table doesn't exist yet
        return pd.DataFrame()


# ---------------------------
# Layout
# ---------------------------
left, right = st.columns([2, 1])

# ---------------------------
# LEFT: Video Upload & Playback
# ---------------------------
with left:
    st.subheader("Upload Video")

    video_file = st.file_uploader(
        "Upload a video file",
        type=["mp4", "avi", "mov"]
    )

    if video_file:
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
        tmp.write(video_file.read())
        st.session_state.video_path = tmp.name

        st.video(st.session_state.video_path)


# ---------------------------
# RIGHT: Controls
# ---------------------------
with right:
    st.subheader("System Controls")

    # ---- START SYSTEM ----
    if st.button("🚀 Start System"):
        if st.session_state.infra_procs or st.session_state.ai_procs:
            st.warning("System already running")
        else:
            st.info("Starting infrastructure services...")
            start_processes(INFRA_COMMANDS, st.session_state.infra_procs)

            time.sleep(1.5)  # allow Flask ports to bind

            st.info("Starting AI services (ZMQ SUBs)...")
            start_processes(AI_SERVICES, st.session_state.ai_procs)

            time.sleep(2.0)  # ZMQ handshake buffer

            st.success("System is ready. Upload a video and run inference.")

    st.divider()
    st.subheader("Detected Vehicles")

    results_placeholder = st.empty()

    df = fetch_reconciled_results()
    if not df.empty:
        results_placeholder.table(df)



    # ---- RUN INFERENCE ----
    if st.button("▶ Run Inference"):
        if not st.session_state.video_path:
            st.error("Please upload a video first")
        elif st.session_state.stream_proc:
            st.warning("Inference already running")
        else:
            st.info("Starting video stream...")
            st.session_state.stream_proc = subprocess.Popen(
                ["python", "stream_handler.py", st.session_state.video_path]
            )
            st.success("Inference started")

    # ---- STOP ALL ----
    if st.button("⏹ Stop All"):
        st.info("Stopping all processes...")

        stop_process(st.session_state.stream_proc)
        st.session_state.stream_proc = None

        stop_processes(st.session_state.ai_procs)
        stop_processes(st.session_state.infra_procs)

        st.success("System stopped cleanly")

    if st.session_state.stream_proc:
        time.sleep(0.5)
        st.rerun()
