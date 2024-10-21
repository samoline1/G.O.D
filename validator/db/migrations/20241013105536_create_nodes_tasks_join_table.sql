-- migrate:up
CREATE TABLE IF NOT EXISTS task_nodes (
    task_id UUID NOT NULL,
    node_id INT NOT NULL,
    PRIMARY KEY (task_id, node_id),
    CONSTRAINT fk_task
        FOREIGN KEY (task_id)
        REFERENCES tasks (task_id)
        ON DELETE CASCADE,
    CONSTRAINT fk_node
        FOREIGN KEY (node_id)
        REFERENCES nodes (node_id)
        ON DELETE CASCADE
);

-- migrate:down
DROP TABLE IF EXISTS task_nodes;
