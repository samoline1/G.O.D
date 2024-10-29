-- migrate:up
-- First, let's ensure we have our nodes table with the original structure
CREATE TABLE IF NOT EXISTS nodes (
    node_id TEXT PRIMARY KEY
);

-- Create nodes_tasks if it doesn't exist
CREATE TABLE IF NOT EXISTS nodes_tasks (
    node_id TEXT,
    task_id INTEGER,
    CONSTRAINT nodes_tasks_node_id_fkey FOREIGN KEY (node_id) REFERENCES nodes(node_id)
);

-- Now we can safely modify the nodes table
-- 1. Add new columns
ALTER TABLE nodes
    ADD COLUMN IF NOT EXISTS hotkey TEXT NOT NULL DEFAULT '',
    ADD COLUMN IF NOT EXISTS incentive FLOAT DEFAULT 0.0,
    ADD COLUMN IF NOT EXISTS netuid INTEGER DEFAULT 0,
    ADD COLUMN IF NOT EXISTS last_updated FLOAT,
    ADD COLUMN IF NOT EXISTS protocol INTEGER DEFAULT 4,
    ADD COLUMN IF NOT EXISTS symmetric_key_uuid TEXT,
    ADD COLUMN IF NOT EXISTS our_validator BOOLEAN DEFAULT FALSE;

-- 2. Update existing rows to have a valid hotkey
UPDATE nodes SET hotkey = node_id WHERE hotkey = '';

-- 3. Drop the foreign key from nodes_tasks
ALTER TABLE nodes_tasks DROP CONSTRAINT IF EXISTS nodes_tasks_node_id_fkey;

-- 4. Modify the primary key
ALTER TABLE nodes DROP CONSTRAINT IF EXISTS nodes_pkey;
ALTER TABLE nodes ADD PRIMARY KEY (hotkey, netuid);

-- 5. Re-create the foreign key constraint to reference hotkey
ALTER TABLE nodes_tasks
    ADD CONSTRAINT nodes_tasks_hotkey_fkey
    FOREIGN KEY (node_id) REFERENCES nodes(hotkey);

-- migrate:down
-- 1. Drop the new foreign key
ALTER TABLE nodes_tasks DROP CONSTRAINT IF EXISTS nodes_tasks_hotkey_fkey;

-- 2. Restore the original primary key
ALTER TABLE nodes DROP CONSTRAINT IF EXISTS nodes_pkey;
ALTER TABLE nodes ADD PRIMARY KEY (node_id);

-- 3. Restore the original foreign key
ALTER TABLE nodes_tasks
    ADD CONSTRAINT nodes_tasks_node_id_fkey
    FOREIGN KEY (node_id) REFERENCES nodes(node_id);

-- 4. Drop the added columns
ALTER TABLE nodes
    DROP COLUMN IF EXISTS hotkey,
    DROP COLUMN IF EXISTS incentive,
    DROP COLUMN IF EXISTS netuid,
    DROP COLUMN IF EXISTS last_updated,
    DROP COLUMN IF EXISTS protocol,
    DROP COLUMN IF EXISTS symmetric_key_uuid,
    DROP COLUMN IF EXISTS our_validator;
