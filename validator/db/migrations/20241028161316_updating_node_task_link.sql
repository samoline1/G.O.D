-- migrate:up
CREATE TABLE IF NOT EXISTS task_nodes (
    task_id UUID NOT NULL,
    hotkey TEXT NOT NULL,
    node_id INT NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (task_id, hotkey, node_id),
);

-- Create index for faster lookups
CREATE INDEX idx_task_nodes_hotkey_netuid ON task_nodes(hotkey, node_id);

-- migrate:down
DROP TABLE IF EXISTS task_nodes;
