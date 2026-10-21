"""
SmartHub database helpers.

All public functions accept an optional *db_path* parameter so that tests
can redirect operations to an isolated temporary database without touching
data/smarthub.db.

Thread/concurrency safety
-------------------------
SQLite is used in WAL mode with a busy timeout.  The ``BEGIN IMMEDIATE``
pattern in borrow/return operations ensures that concurrent writers queue
rather than fail immediately.  ``get_connection()`` sets a 10-second busy
timeout so that brief lock contention does not cause spurious errors.
"""

import logging
import sqlite3
from pathlib import Path
from typing import Optional, Sequence

logger = logging.getLogger(__name__)

# Project root: SmartHub/
BASE_DIR = Path(__file__).resolve().parents[2]

# Default SQLite database location (never used during tests)
DATABASE_PATH = BASE_DIR / "data" / "smarthub.db"

# SQL schema location
SCHEMA_PATH = BASE_DIR / "app" / "database" / "schema.sql"

# Valid final statuses for a session
_VALID_SESSION_STATUSES = frozenset({"completed", "cancelled", "failed"})

# Valid transaction statuses
_VALID_TRANSACTION_STATUSES = frozenset({"pending", "completed", "failed", "refunded"})

# Valid payment methods accepted at the kiosk
_VALID_PAYMENT_METHODS = frozenset({"coin", "mock", "free", "card"})

# Maximum RFID UID length (prevent oversized inputs)
_MAX_RFID_LEN = 64

# After this many hours an 'active' session is considered stale
_STALE_SESSION_HOURS = 8


# ---------------------------------------------------------------------------
# Connection helpers
# ---------------------------------------------------------------------------


def get_connection(db_path: Optional[Path] = None) -> sqlite3.Connection:
    """Return a connection to the SmartHub SQLite database.

    Foreign-key enforcement is turned on for every connection.  The
    ``row_factory`` is set so that rows behave like dicts.  A 10-second
    busy timeout prevents immediate failure under brief lock contention.
    WAL mode is enabled for better concurrent read/write throughput.
    """
    path = Path(db_path) if db_path else DATABASE_PATH
    connection = sqlite3.connect(str(path), timeout=10)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA foreign_keys = ON")
    connection.execute("PRAGMA journal_mode = WAL")
    return connection


def initialize_database(db_path: Optional[Path] = None) -> None:
    """Create database tables using schema.sql.

    Safe to call on an already-initialised database — uses IF NOT EXISTS
    and INSERT OR IGNORE for seed data.
    """
    path = Path(db_path) if db_path else DATABASE_PATH
    path.parent.mkdir(parents=True, exist_ok=True)

    connection = get_connection(path)
    try:
        schema = SCHEMA_PATH.read_text(encoding="utf-8")
        connection.executescript(schema)
        connection.commit()
        logger.info("Database initialised at %s", path)
    finally:
        connection.close()


# ---------------------------------------------------------------------------
# Low-level query helpers
# ---------------------------------------------------------------------------


def execute_query(
    query: str,
    parameters: Sequence = (),
    db_path: Optional[Path] = None,
) -> list[sqlite3.Row]:
    """Execute a SELECT query and return the resulting rows."""
    connection = get_connection(db_path)
    try:
        cursor = connection.execute(query, parameters)
        return cursor.fetchall()
    except sqlite3.Error:
        logger.exception("execute_query failed: %s", query[:80])
        raise
    finally:
        connection.close()


def execute_insert(
    query: str,
    parameters: Sequence = (),
    db_path: Optional[Path] = None,
) -> int:
    """Execute an INSERT query and return the new row ID."""
    connection = get_connection(db_path)
    try:
        cursor = connection.execute(query, parameters)
        connection.commit()
        return cursor.lastrowid
    except sqlite3.Error:
        connection.rollback()
        raise
    finally:
        connection.close()


# ---------------------------------------------------------------------------
# Input validation helpers
# ---------------------------------------------------------------------------


def _validate_rfid(rfid_uid: str, label: str = "RFID UID") -> str:
    """Strip and validate an RFID UID string.  Returns the stripped value."""
    if not rfid_uid or not rfid_uid.strip():
        raise ValueError(f"{label} cannot be empty")
    stripped = rfid_uid.strip()
    if len(stripped) > _MAX_RFID_LEN:
        raise ValueError(
            f"{label} is too long (max {_MAX_RFID_LEN} characters)"
        )
    return stripped


def _validate_limit(limit: int) -> None:
    """Raise ValueError if *limit* is not a positive integer."""
    if not isinstance(limit, int) or isinstance(limit, bool) or limit < 1:
        raise ValueError("limit must be a positive integer")


# ---------------------------------------------------------------------------
# User management
# ---------------------------------------------------------------------------


def create_user(rfid_uid: str, db_path: Optional[Path] = None) -> Optional[int]:
    """Create a user identified by an RFID UID.

    Returns the new user ID, or *None* if the RFID UID is already
    registered.

    Raises ValueError for empty, whitespace-only, or excessively long UIDs.
    """
    uid = _validate_rfid(rfid_uid)
    query = "INSERT INTO users (rfid_uid) VALUES (?)"
    try:
        row_id = execute_insert(query, (uid,), db_path)
        logger.debug("Created user id=%s rfid=%s", row_id, uid)
        return row_id
    except sqlite3.IntegrityError:
        logger.debug("Duplicate RFID registration attempt: %s", uid)
        return None


def get_user_by_rfid(
    rfid_uid: str, db_path: Optional[Path] = None
) -> Optional[sqlite3.Row]:
    """Return the user row matching *rfid_uid*, or *None*."""
    query = "SELECT id, rfid_uid, created_at FROM users WHERE rfid_uid = ?"
    rows = execute_query(query, (rfid_uid,), db_path)
    return rows[0] if rows else None


def get_user_by_id(
    user_id: int, db_path: Optional[Path] = None
) -> Optional[sqlite3.Row]:
    """Return the user row with the given *user_id*, or *None*."""
    query = "SELECT id, rfid_uid, created_at FROM users WHERE id = ?"
    rows = execute_query(query, (user_id,), db_path)
    return rows[0] if rows else None


# ---------------------------------------------------------------------------
# Session management
# ---------------------------------------------------------------------------


def create_session(
    user_id: int,
    service_name: str,
    db_path: Optional[Path] = None,
) -> int:
    """Create an active session for CHARGE, STORE, or BORROW.

    Raises ValueError if *service_name* is not one of the known services
    or if no service row exists in the database.
    """
    if not service_name or not service_name.strip():
        raise ValueError("Service name cannot be empty")

    normalized = service_name.strip().upper()
    query = """
        INSERT INTO sessions (user_id, service_id, status)
        VALUES (
            ?,
            (SELECT id FROM services WHERE name = ?),
            'active'
        )
    """
    connection = get_connection(db_path)
    try:
        try:
            with connection:
                cursor = connection.execute(query, (user_id, normalized))
                if cursor.lastrowid is None or cursor.rowcount == 0:
                    raise ValueError(f"Unknown service: {service_name!r}")

                # Verify service_id was actually resolved (not NULL due to
                # subquery returning no rows)
                row = connection.execute(
                    "SELECT service_id FROM sessions WHERE id = ?",
                    (cursor.lastrowid,),
                ).fetchone()
                if row is None or row["service_id"] is None:
                    raise ValueError(f"Unknown service: {service_name!r}")

                session_id = cursor.lastrowid
                logger.debug(
                    "Created session id=%s user_id=%s service=%s",
                    session_id,
                    user_id,
                    normalized,
                )
                return session_id
        except sqlite3.IntegrityError as exc:
            # NULL service_id means the sub-SELECT found no matching service
            raise ValueError(f"Unknown service: {service_name!r}") from exc
    finally:
        connection.close()


def get_session(
    session_id: int, db_path: Optional[Path] = None
) -> Optional[sqlite3.Row]:
    """Return a session row by ID, or *None*."""
    query = """
        SELECT s.id, s.user_id, s.service_id, sv.name AS service_name,
               s.started_at, s.ended_at, s.duration_seconds, s.status
        FROM sessions AS s
        JOIN services AS sv ON s.service_id = sv.id
        WHERE s.id = ?
    """
    rows = execute_query(query, (session_id,), db_path)
    return rows[0] if rows else None


def get_active_sessions_for_user(
    user_id: int,
    service_name: str,
    db_path: Optional[Path] = None,
) -> list[sqlite3.Row]:
    """Return all currently active sessions for *user_id* in *service_name*."""
    query = """
        SELECT s.id, s.user_id, sv.name AS service_name, s.started_at, s.status
        FROM sessions AS s
        JOIN services AS sv ON s.service_id = sv.id
        WHERE s.user_id = ?
          AND sv.name = ?
          AND s.status = 'active'
        ORDER BY s.id DESC
    """
    return execute_query(query, (user_id, service_name.upper()), db_path)


def count_active_sessions_by_service(
    service_name: str, db_path: Optional[Path] = None
) -> int:
    """Return the number of active sessions for a service."""
    rows = execute_query(
        """
        SELECT COUNT(*) AS session_count
        FROM sessions AS s JOIN services AS sv ON s.service_id = sv.id
        WHERE sv.name = ? AND s.status = 'active'
        """,
        (service_name.strip().upper(),),
        db_path,
    )
    return int(rows[0]["session_count"])


def end_session(
    session_id: int,
    status: str = "completed",
    db_path: Optional[Path] = None,
) -> bool:
    """End a session and calculate its duration.

    Returns True if the session was successfully ended, False if the
    session was already ended or does not exist.

    Raises ValueError for unsupported final statuses.
    """
    if status not in _VALID_SESSION_STATUSES:
        raise ValueError(
            f"Invalid final session status {status!r}. "
            f"Expected one of: {sorted(_VALID_SESSION_STATUSES)}"
        )

    query = """
        UPDATE sessions
        SET ended_at = CURRENT_TIMESTAMP,
            duration_seconds = CAST(
                (julianday(CURRENT_TIMESTAMP) - julianday(started_at))
                * 86400 AS INTEGER
            ),
            status = ?
        WHERE id = ? AND status = 'active'
    """
    connection = get_connection(db_path)
    try:
        with connection:
            cursor = connection.execute(query, (status, session_id))
            changed = cursor.rowcount == 1
            if changed:
                logger.debug("Ended session id=%s status=%s", session_id, status)
            else:
                logger.debug(
                    "end_session: session %s not active or not found", session_id
                )
            return changed
    finally:
        connection.close()


def timeout_stale_sessions(
    hours: int = _STALE_SESSION_HOURS,
    db_path: Optional[Path] = None,
) -> int:
    """Mark sessions that have been 'active' for more than *hours* as 'failed'.

    This is a recovery mechanism for sessions left open after a crash or
    power loss.  Returns the number of sessions that were closed.

    Should be called at application startup and/or on a periodic schedule.
    """
    if not isinstance(hours, int) or hours < 1:
        raise ValueError("hours must be a positive integer")

    query = """
        UPDATE sessions
        SET ended_at = CURRENT_TIMESTAMP,
            duration_seconds = CAST(
                (julianday(CURRENT_TIMESTAMP) - julianday(started_at))
                * 86400 AS INTEGER
            ),
            status = 'failed'
        WHERE status = 'active'
          AND julianday(CURRENT_TIMESTAMP) - julianday(started_at) > ?
    """
    days_threshold = hours / 24.0
    connection = get_connection(db_path)
    try:
        with connection:
            cursor = connection.execute(query, (days_threshold,))
            count = cursor.rowcount
            if count > 0:
                logger.warning("Timed out %d stale session(s)", count)
            return count
    finally:
        connection.close()


def get_sessions_by_service(
    service_name: str,
    limit: int = 50,
    db_path: Optional[Path] = None,
) -> list[sqlite3.Row]:
    """Return recent sessions for a given service name."""
    _validate_limit(limit)
    query = """
        SELECT s.id, s.user_id, sv.name AS service_name,
               s.started_at, s.ended_at, s.duration_seconds, s.status
        FROM sessions AS s
        JOIN services AS sv ON s.service_id = sv.id
        WHERE sv.name = ?
        ORDER BY s.id DESC
        LIMIT ?
    """
    return execute_query(query, (service_name.upper(), limit), db_path)


# ---------------------------------------------------------------------------
# Transactions
# ---------------------------------------------------------------------------


def record_transaction(
    amount: float,
    session_id: Optional[int] = None,
    payment_method: str = "coin",
    status: str = "completed",
    db_path: Optional[Path] = None,
) -> int:
    """Record a payment transaction.

    Raises ValueError for negative amounts, unsupported statuses, or
    unrecognised payment methods.
    """
    if not isinstance(amount, (int, float)) or isinstance(amount, bool):
        raise ValueError("Transaction amount must be a number")
    if amount < 0:
        raise ValueError("Transaction amount cannot be negative")
    if not payment_method or not payment_method.strip():
        raise ValueError("Payment method cannot be empty")
    pm = payment_method.strip().lower()
    if pm not in _VALID_PAYMENT_METHODS:
        raise ValueError(
            f"Invalid payment method {payment_method!r}. "
            f"Expected one of: {sorted(_VALID_PAYMENT_METHODS)}"
        )
    if status not in _VALID_TRANSACTION_STATUSES:
        raise ValueError(
            f"Invalid transaction status {status!r}. "
            f"Expected one of: {sorted(_VALID_TRANSACTION_STATUSES)}"
        )

    query = """
        INSERT INTO transactions (session_id, amount, payment_method, status)
        VALUES (?, ?, ?, ?)
    """
    return execute_insert(query, (session_id, amount, pm, status), db_path)


def get_transactions(
    limit: int = 50,
    session_id: Optional[int] = None,
    db_path: Optional[Path] = None,
) -> list[sqlite3.Row]:
    """Return recent transactions, newest first.

    If *session_id* is provided, only transactions for that session are
    returned.
    """
    _validate_limit(limit)
    if session_id is not None:
        query = """
            SELECT
                t.id, t.session_id, t.amount, t.payment_method,
                t.status, t.created_at, s.user_id,
                sv.name AS service_name
            FROM transactions AS t
            LEFT JOIN sessions AS s ON t.session_id = s.id
            LEFT JOIN services AS sv ON s.service_id = sv.id
            WHERE t.session_id = ?
            ORDER BY t.id DESC
            LIMIT ?
        """
        return execute_query(query, (session_id, limit), db_path)

    query = """
        SELECT
            t.id, t.session_id, t.amount, t.payment_method,
            t.status, t.created_at, s.user_id,
            sv.name AS service_name
        FROM transactions AS t
        LEFT JOIN sessions AS s ON t.session_id = s.id
        LEFT JOIN services AS sv ON s.service_id = sv.id
        ORDER BY t.id DESC
        LIMIT ?
    """
    return execute_query(query, (limit,), db_path)


# ---------------------------------------------------------------------------
# Items
# ---------------------------------------------------------------------------


def create_item(
    rfid_uid: str,
    item_type: str,
    db_path: Optional[Path] = None,
) -> Optional[int]:
    """Register a borrowable item using its RFID UID.

    Returns the new item ID, or *None* if the RFID UID is already
    registered.
    """
    uid = _validate_rfid(rfid_uid, label="Item RFID UID")
    if not item_type or not item_type.strip():
        raise ValueError("Item type cannot be empty")
    it = item_type.strip()
    if len(it) > 64:
        raise ValueError("Item type is too long (max 64 characters)")

    query = "INSERT INTO items (rfid_uid, item_type, status) VALUES (?, ?, 'available')"
    try:
        return execute_insert(query, (uid, it), db_path)
    except sqlite3.IntegrityError:
        logger.debug("Duplicate item RFID: %s", rfid_uid)
        return None


def get_available_items(db_path: Optional[Path] = None) -> list[sqlite3.Row]:
    """Return all items currently available to borrow."""
    query = """
        SELECT id, rfid_uid, item_type, status, created_at
        FROM items
        WHERE status = 'available'
        ORDER BY id
    """
    return execute_query(query, (), db_path)


def get_item_by_rfid(
    rfid_uid: str, db_path: Optional[Path] = None
) -> Optional[sqlite3.Row]:
    """Return the item row matching *rfid_uid*, or *None*."""
    query = "SELECT id, rfid_uid, item_type, status FROM items WHERE rfid_uid = ?"
    rows = execute_query(query, (rfid_uid,), db_path)
    return rows[0] if rows else None


def get_item_by_id(
    item_id: int, db_path: Optional[Path] = None
) -> Optional[sqlite3.Row]:
    """Return the item row with the given *item_id*, or *None*."""
    query = "SELECT id, rfid_uid, item_type, status, created_at FROM items WHERE id = ?"
    rows = execute_query(query, (item_id,), db_path)
    return rows[0] if rows else None


# ---------------------------------------------------------------------------
# Borrow / return
# ---------------------------------------------------------------------------


def record_borrow(
    session_id: int,
    item_id: int,
    db_path: Optional[Path] = None,
) -> int:
    """Record that an item has been borrowed.

    Atomically checks item availability, validates the session, creates the
    borrow record, and marks the item as borrowed.

    Raises ValueError if the item is unavailable, does not exist, or the
    session is not active.
    """
    connection = get_connection(db_path)
    try:
        connection.execute("BEGIN IMMEDIATE")

        item = connection.execute(
            "SELECT id, status FROM items WHERE id = ?", (item_id,)
        ).fetchone()
        if item is None:
            raise ValueError(f"Item id={item_id} does not exist")
        if item["status"] != "available":
            raise ValueError(
                f"Item id={item_id} is not available (status={item['status']!r})"
            )

        session = connection.execute(
            "SELECT id, status FROM sessions WHERE id = ?", (session_id,)
        ).fetchone()
        if session is None:
            raise ValueError(f"Session id={session_id} does not exist")
        if session["status"] != "active":
            raise ValueError(
                f"Session id={session_id} is not active (status={session['status']!r})"
            )

        cursor = connection.execute(
            "INSERT INTO borrow_records (session_id, item_id, status) VALUES (?, ?, 'borrowed')",
            (session_id, item_id),
        )
        connection.execute(
            "UPDATE items SET status = 'borrowed' WHERE id = ?", (item_id,)
        )
        connection.commit()
        borrow_id = cursor.lastrowid
        logger.debug(
            "Borrow recorded id=%s session=%s item=%s", borrow_id, session_id, item_id
        )
        return borrow_id
    except Exception:
        connection.rollback()
        raise
    finally:
        connection.close()


def record_return(item_id: int, db_path: Optional[Path] = None) -> bool:
    """Record an item's return and make it available again.

    Atomically finds the active borrow record, closes it, and resets the
    item status to 'available'.

    Raises ValueError if no active borrow record exists.
    """
    connection = get_connection(db_path)
    try:
        connection.execute("BEGIN IMMEDIATE")

        borrow = connection.execute(
            """
            SELECT id FROM borrow_records
            WHERE item_id = ? AND status = 'borrowed'
            ORDER BY id DESC LIMIT 1
            """,
            (item_id,),
        ).fetchone()
        if borrow is None:
            raise ValueError(f"No active borrow record for item id={item_id}")

        connection.execute(
            """
            UPDATE borrow_records
            SET returned_at = CURRENT_TIMESTAMP, status = 'returned'
            WHERE id = ?
            """,
            (borrow["id"],),
        )
        connection.execute(
            "UPDATE items SET status = 'available' WHERE id = ?", (item_id,)
        )
        connection.commit()
        logger.debug("Return recorded for item id=%s", item_id)
        return True
    except Exception:
        connection.rollback()
        raise
    finally:
        connection.close()


def record_return_and_end_session(
    item_id: int, db_path: Optional[Path] = None
) -> int:
    """Atomically return an item and complete the linked BORROW session.

    The older :func:`record_return` remains available for compatibility.
    This helper is used by the service workflow, where inventory and session
    state must never be committed independently.
    """
    connection = get_connection(db_path)
    try:
        connection.execute("BEGIN IMMEDIATE")
        borrow = connection.execute(
            """
            SELECT id, session_id FROM borrow_records
            WHERE item_id = ? AND status = 'borrowed'
            ORDER BY id DESC LIMIT 1
            """,
            (item_id,),
        ).fetchone()
        if borrow is None or borrow["session_id"] is None:
            raise ValueError(f"No active borrow record for item id={item_id}")

        changed = connection.execute(
            """
            UPDATE sessions
            SET ended_at = CURRENT_TIMESTAMP,
                duration_seconds = CAST((julianday(CURRENT_TIMESTAMP) - julianday(started_at)) * 86400 AS INTEGER),
                status = 'completed'
            WHERE id = ? AND status = 'active'
            """,
            (borrow["session_id"],),
        ).rowcount
        if changed != 1:
            raise ValueError(f"Borrow session id={borrow['session_id']} is not active")

        connection.execute(
            "UPDATE borrow_records SET returned_at = CURRENT_TIMESTAMP, status = 'returned' WHERE id = ?",
            (borrow["id"],),
        )
        connection.execute("UPDATE items SET status = 'available' WHERE id = ?", (item_id,))
        connection.commit()
        return int(borrow["session_id"])
    except Exception:
        connection.rollback()
        raise
    finally:
        connection.close()


def get_borrow_records(
    limit: int = 50, db_path: Optional[Path] = None
) -> list[sqlite3.Row]:
    """Return recent borrow records, newest first."""
    _validate_limit(limit)
    query = """
        SELECT
            br.id, br.session_id, br.item_id,
            i.rfid_uid AS item_rfid, i.item_type,
            br.borrowed_at, br.returned_at, br.status
        FROM borrow_records AS br
        JOIN items AS i ON br.item_id = i.id
        ORDER BY br.id DESC
        LIMIT ?
    """
    return execute_query(query, (limit,), db_path)


# ---------------------------------------------------------------------------
# System events
# ---------------------------------------------------------------------------


def log_system_event(
    event_type: str,
    service: Optional[str] = None,
    message: Optional[str] = None,
    db_path: Optional[Path] = None,
) -> int:
    """Record a SmartHub system event."""
    if not event_type or not event_type.strip():
        raise ValueError("Event type cannot be empty")

    query = """
        INSERT INTO system_events (event_type, service, message)
        VALUES (?, ?, ?)
    """
    return execute_insert(
        query,
        (
            event_type.strip(),
            service.strip() if service else None,
            message.strip() if message else None,
        ),
        db_path,
    )


def get_recent_system_events(
    limit: int = 50, db_path: Optional[Path] = None
) -> list[sqlite3.Row]:
    """Return recent system events, newest first."""
    _validate_limit(limit)
    query = """
        SELECT id, event_type, service, message, created_at
        FROM system_events
        ORDER BY id DESC
        LIMIT ?
    """
    return execute_query(query, (limit,), db_path)


# ---------------------------------------------------------------------------
# Dashboard / metrics helpers
# ---------------------------------------------------------------------------


def get_dashboard_summary(db_path: Optional[Path] = None) -> dict:
    """Return a dict with aggregate counts for the dashboard."""
    connection = get_connection(db_path)
    try:
        total_sessions = connection.execute(
            "SELECT COUNT(*) FROM sessions"
        ).fetchone()[0]
        active_sessions = connection.execute(
            "SELECT COUNT(*) FROM sessions WHERE status = 'active'"
        ).fetchone()[0]
        total_users = connection.execute(
            "SELECT COUNT(*) FROM users"
        ).fetchone()[0]
        total_revenue = connection.execute(
            "SELECT COALESCE(SUM(amount), 0) FROM transactions WHERE status = 'completed'"
        ).fetchone()[0]
        available_items = connection.execute(
            "SELECT COUNT(*) FROM items WHERE status = 'available'"
        ).fetchone()[0]
        borrowed_items = connection.execute(
            "SELECT COUNT(*) FROM items WHERE status = 'borrowed'"
        ).fetchone()[0]

        # Per-service session counts
        service_rows = connection.execute(
            """
            SELECT sv.name, COUNT(s.id) AS session_count
            FROM services AS sv
            LEFT JOIN sessions AS s ON sv.id = s.service_id
            GROUP BY sv.id
            """
        ).fetchall()
        service_counts = {row["name"]: row["session_count"] for row in service_rows}

        return {
            "total_sessions": total_sessions,
            "active_sessions": active_sessions,
            "total_users": total_users,
            "total_revenue": round(float(total_revenue), 2),
            "available_items": available_items,
            "borrowed_items": borrowed_items,
            "service_counts": service_counts,
        }
    finally:
        connection.close()


def get_usage_metrics(
    limit: int = 100, db_path: Optional[Path] = None
) -> list[sqlite3.Row]:
    """Return recent completed sessions with duration for usage analysis."""
    _validate_limit(limit)
    query = """
        SELECT
            s.id, sv.name AS service_name,
            s.started_at, s.ended_at, s.duration_seconds, s.status
        FROM sessions AS s
        JOIN services AS sv ON s.service_id = sv.id
        WHERE s.status IN ('completed', 'cancelled', 'failed')
        ORDER BY s.id DESC
        LIMIT ?
    """
    return execute_query(query, (limit,), db_path)
