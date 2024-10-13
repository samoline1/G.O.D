-- migrate:up
CREATE TABLE IF NOT EXISTS nodes (
    node_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    coldkey TEXT NOT NULL,
    ip TEXT NOT NULL,
    ip_type TEXT NOT NULL,
    port INTEGER NOT NULL,
    symmetric_key TEXT NOT NULL,
    network FLOAT NOT NULL,
    trust FLOAT NOT NULL,
    vtrust FLOAT NOT NULL,
    stake FLOAT NOT NULL,
    created_timestamp TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
    updated_timestamp TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP
);

-- migrate:down
DROP TABLE IF EXISTS nodes;
