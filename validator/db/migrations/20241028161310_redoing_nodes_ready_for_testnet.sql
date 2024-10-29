-- migrate:up
-- First drop the foreign key constraints
ALTER TABLE nodes_tasks DROP CONSTRAINT IF EXISTS nodes_tasks_node_id_fkey;

-- Add the new columns
ALTER TABLE nodes
    ADD COLUMN IF NOT EXISTS hotkey TEXT NOT NULL DEFAULT '',
    ADD COLUMN IF NOT EXISTS incentive FLOAT DEFAULT 0.0,
    ADD COLUMN IF NOT EXISTS netuid INTEGER DEFAULT 0,
    ADD COLUMN IF NOT EXISTS last_updated FLOAT,
    ADD COLUMN IF NOT EXISTS protocol INTEGER DEFAULT 4,
    ADD COLUMN IF NOT EXISTS symmetric_key_uuid TEXT,
    ADD COLUMN IF NOT EXISTS our_validator BOOLEAN DEFAULT FALSE;

-- Update existing rows to have a valid hotkey
UPDATE nodes SET hotkey = node_id WHERE hotkey = '';

-- Now we can safely modify the primary key
ALTER TABLE nodes DROP CONSTRAINT IF EXISTS nodes_pkey;
ALTER TABLE nodes ADD PRIMARY KEY (hotkey, netuid);

-- Re-create the foreign key constraint to reference the correct column
ALTER TABLE nodes_tasks
    ADD CONSTRAINT nodes_tasks_hotkey_fkey
    FOREIGN KEY (node_id) REFERENCES nodes(hotkey);

-- migrate:down
-- First drop the new foreign key
ALTER TABLE nodes_tasks DROP CONSTRAINT IF EXISTS nodes_tasks_hotkey_fkey;

-- Restore the original primary key
ALTER TABLE nodes DROP CONSTRAINT IF EXISTS nodes_pkey;
ALTER TABLE nodes ADD PRIMARY KEY (node_id);

-- Restore the original foreign key
ALTER TABLE nodes_tasks
    ADD CONSTRAINT nodes_tasks_node_id_fkey
    FOREIGN KEY (node_id) REFERENCES nodes(node_id);

-- Drop the added columns
ALTER TABLE nodes
    DROP COLUMN IF EXISTS hotkey,
    DROP COLUMN IF EXISTS incentive,
    DROP COLUMN IF EXISTS netuid,
    DROP COLUMN IF EXISTS last_updated,
    DROP COLUMN IF EXISTS protocol,
    DROP COLUMN IF EXISTS symmetric_key_uuid,
    DROP COLUMN IF EXISTS our_validator;
