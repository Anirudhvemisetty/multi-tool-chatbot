import sqlite3
import bcrypt

DB_PATH = "chatbot.db"

conn = sqlite3.connect(
    DB_PATH,
    check_same_thread=False,
)

conn.execute("""
CREATE TABLE IF NOT EXISTS users (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    username TEXT UNIQUE NOT NULL,
    password_hash TEXT NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
)
""")

conn.execute("""
CREATE TABLE IF NOT EXISTS user_threads (
    thread_id TEXT PRIMARY KEY,
    user_id INTEGER NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,

    FOREIGN KEY(user_id)
    REFERENCES users(id)
)
""")

conn.commit()


def signup_user(
    username,
    password,
):

    if not username.strip():
        return False, "Username required"

    if len(password) < 6:
        return False, "Password must be at least 6 characters"

    try:

        password_hash = bcrypt.hashpw(
            password.encode(),
            bcrypt.gensalt(),
        ).decode()

        conn.execute(
            """
            INSERT INTO users
            (
                username,
                password_hash
            )
            VALUES (?, ?)
            """,
            (
                username,
                password_hash,
            ),
        )

        conn.commit()

        return True, "Signup successful"

    except sqlite3.IntegrityError:

        return False, "Username already exists"
    


def login_user(
    username,
    password,
):

    cursor = conn.execute(
        """
        SELECT
            id,
            password_hash
        FROM users
        WHERE username = ?
        """,
        (username,),
    )

    user = cursor.fetchone()

    if not user:
        return None

    user_id = user[0]
    stored_hash = user[1]

    if bcrypt.checkpw(
        password.encode(),
        stored_hash.encode(),
    ):
        return user_id

    return None

def create_thread_for_user(
    thread_id,
    user_id,
):

    conn.execute(
        """
        INSERT OR IGNORE
        INTO user_threads
        (
            thread_id,
            user_id
        )
        VALUES (?, ?)
        """,
        (
            thread_id,
            user_id,
        ),
    )

    conn.commit()
    
def delete_thread(
    thread_id,
    user_id,
):

    conn.execute(
        """
        DELETE FROM user_threads
        WHERE thread_id = ?
        AND user_id = ?
        """,
        (
            thread_id,
            user_id,
        ),
    )

    conn.commit()

   

def retrieve_user_threads(
    user_id,
):

    cursor = conn.execute(
        """
        SELECT thread_id
        FROM user_threads
        WHERE user_id = ?
        ORDER BY created_at DESC
        """,
        (user_id,),
    )

    return [
        row[0]
        for row in cursor.fetchall()
    ]