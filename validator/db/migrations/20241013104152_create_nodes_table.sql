-- migrate:up
CREATE TABLE IF NOT EXISTS nodes (
    node_id INT PRIMARY KEY DEFAULT 0,
    coldkey TEXT NOT NULL,
    ip TEXT NOT NULL,
    ip_type TEXT NOT NULL,
    port INTEGER NOT NULL,
    symmetric_key TEXT NOT NULL,
    network FLOAT NOT NULL,
    trust FLOAT,
    vtrust FLOAT,
    stake FLOAT NOT NULL,
    created_timestamp TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
    updated_timestamp TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP
);

-- migrate:down
DROP TABLE IF EXISTS nodes;
