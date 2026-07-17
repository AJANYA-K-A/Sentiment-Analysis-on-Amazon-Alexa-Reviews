"""
migrate_db.py
Run this ONCE to add the new columns to your existing sentiscope.db.
    python migrate_db.py
"""
import sqlite3
import os

DB_PATH = os.path.join(os.path.dirname(__file__), "instance", "sentiscope.db")

def column_exists(cursor, table, column):
    cursor.execute(f"PRAGMA table_info({table})")
    return any(row[1] == column for row in cursor.fetchall())

def run():
    conn = sqlite3.connect(DB_PATH)
    cur  = conn.cursor()

    # ── users table ──────────────────────────────────────────────────────────
    if not column_exists(cur, "users", "bio"):
        cur.execute("ALTER TABLE users ADD COLUMN bio VARCHAR(300) DEFAULT ''")
        print("✓ Added users.bio")
    else:
        print("– users.bio already exists")

    if not column_exists(cur, "users", "created_at"):
        cur.execute("ALTER TABLE users ADD COLUMN created_at DATETIME DEFAULT CURRENT_TIMESTAMP")
        print("✓ Added users.created_at")
    else:
        print("– users.created_at already exists")

    # ── analysis_history table (create if missing, add cols if partial) ──────
    cur.execute("""
        CREATE TABLE IF NOT EXISTS analysis_history (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id     INTEGER NOT NULL REFERENCES users(id),
            text        TEXT    NOT NULL,
            label       VARCHAR(20) NOT NULL,
            confidence  REAL    NOT NULL,
            vader_score REAL    NOT NULL,
            tb_score    REAL    NOT NULL,
            mode        VARCHAR(20) DEFAULT 'single',
            created_at  DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    """)
    print("✓ analysis_history table ready")

    conn.commit()
    conn.close()
    print("\nMigration complete. You can now run app.py normally.")

if __name__ == "__main__":
    run()