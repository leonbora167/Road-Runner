import sqlite3
import time
from collections import Counter

DB_PATH = "data/ocr_results.db"
CONF_THRESH = 0.60
MIN_SAMPLES = 5


def normalize(text: str) -> str:
    return text.upper().replace(" ", "")


def reconcile():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    # Find vehicle IDs that are NOT yet reconciled
    cursor.execute("""
        SELECT DISTINCT vehicle_track_id
        FROM ocr_results
        WHERE vehicle_track_id NOT IN (
            SELECT vehicle_track_id FROM vehicle_plates
        )
    """)

    vehicle_ids = [r[0] for r in cursor.fetchall()]

    for vid in vehicle_ids:
        cursor.execute("""
            SELECT rec_text, rec_confidence
            FROM ocr_results
            WHERE vehicle_track_id = ?
              AND rec_confidence >= ?
        """, (vid, CONF_THRESH))

        rows = cursor.fetchall()

        if len(rows) < MIN_SAMPLES:
            continue

        texts = [normalize(r[0]) for r in rows if len(r[0]) >= 6]

        if not texts:
            continue

        counter = Counter(texts)
        final_plate, votes = counter.most_common(1)[0]

        avg_conf = sum(r[1] for r in rows) / len(rows)

        cursor.execute("""
            INSERT INTO vehicle_plates (
                vehicle_track_id,
                final_plate,
                vote_count,
                avg_confidence,
                resolved_at
            ) VALUES (?, ?, ?, ?, ?)
        """, (
            vid,
            final_plate,
            votes,
            avg_conf,
            time.time()
        ))

        conn.commit()
        #print(f"[RECONCILE] Vehicle {vid} resolved as {final_plate}") -> TO implement as log later

    conn.close()


if __name__ == "__main__":
    reconcile()
