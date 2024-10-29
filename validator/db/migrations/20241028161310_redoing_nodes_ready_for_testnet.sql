-- migrate:up
-- Add new columns to nodes table
ALTER TABLE nodes
    ADD COLUMN IF NOT EXISTS hotkey TEXT,
    ADD COLUMN IF NOT EXISTS incentive FLOAT DEFAULT 0.0,
    ADD COLUMN IF NOT EXISTS netuid INTEGER DEFAULT 0,
    ADD COLUMN IF NOT EXISTS last_updated FLOAT,
    ADD COLUMN IF NOT EXISTS protocol INTEGER DEFAULT 4,
    ADD COLUMN IF NOT EXISTS symmetric_key_uuid TEXT,
    ADD COLUMN IF NOT EXISTS our_validator BOOLEAN DEFAULT FALSE;
    ALTER COLUMN ip_type TYPE INTEGER USING ip_type::INTEGER;

-- Create indexes for frequently queried columns
CREATE INDEX IF NOT EXISTS idx_nodes_hotkey ON nodes(hotkey);
CREATE INDEX IF NOT EXISTS idx_nodes_netuid ON nodes(netuid);

-- migrate:down
-- Remove added columns
ALTER TABLE nodes
    DROP COLUMN IF EXISTS hotkey,
    DROP COLUMN IF EXISTS incentive,
    DROP COLUMN IF EXISTS netuid,
    DROP COLUMN IF EXISTS last_updated,
    DROP COLUMN IF EXISTS protocol,
    DROP COLUMN IF EXISTS symmetric_key_uuid,
    DROP COLUMN IF EXISTS our_validator;

-- Remove the indexes
DROP INDEX IF EXISTS idx_nodes_hotkey;
DROP INDEX IF EXISTS idx_nodes_netuid;
