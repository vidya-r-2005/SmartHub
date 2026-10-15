PRAGMA foreign_keys = ON;

-- ============================================
-- SmartHub Database Schema
-- ============================================

-- Users identified by RFID
CREATE TABLE IF NOT EXISTS users (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    rfid_uid TEXT NOT NULL UNIQUE,
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);

-- SmartHub services
CREATE TABLE IF NOT EXISTS services (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL UNIQUE,
    description TEXT
);

-- Every CHARGE / STORE / BORROW session
CREATE TABLE IF NOT EXISTS sessions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id INTEGER,
    service_id INTEGER NOT NULL,
    started_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    ended_at TEXT,
    duration_seconds INTEGER,
    status TEXT NOT NULL DEFAULT 'active',

    FOREIGN KEY (user_id) REFERENCES users(id),
    FOREIGN KEY (service_id) REFERENCES services(id)
);

-- Payment / coin information
CREATE TABLE IF NOT EXISTS transactions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id INTEGER,
    amount REAL NOT NULL DEFAULT 0,
    payment_method TEXT NOT NULL DEFAULT 'coin',
    status TEXT NOT NULL DEFAULT 'completed',
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,

    FOREIGN KEY (session_id) REFERENCES sessions(id)
);

-- Items that can be borrowed
CREATE TABLE IF NOT EXISTS items (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    rfid_uid TEXT NOT NULL UNIQUE,
    item_type TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'available',
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);

-- Borrow / return records
CREATE TABLE IF NOT EXISTS borrow_records (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id INTEGER,
    item_id INTEGER NOT NULL,
    borrowed_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    returned_at TEXT,
    status TEXT NOT NULL DEFAULT 'borrowed',

    FOREIGN KEY (session_id) REFERENCES sessions(id),
    FOREIGN KEY (item_id) REFERENCES items(id)
);

-- General system events
CREATE TABLE IF NOT EXISTS system_events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    event_type TEXT NOT NULL,
    service TEXT,
    message TEXT,
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);

-- ============================================
-- Initial SmartHub services
-- ============================================

INSERT OR IGNORE INTO services (name, description)
VALUES
    ('CHARGE', 'Paid/timed device charging'),
    ('STORE', 'Secure temporary storage'),
    ('BORROW', 'Power-bank/accessory borrowing');