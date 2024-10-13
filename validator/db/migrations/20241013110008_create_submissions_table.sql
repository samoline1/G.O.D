-- migrate:up
CREATE TABLE IF NOT EXISTS submissions (
    submission_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    task_id UUID NOT NULL,
    node_id UUID NOT NULL,
    repo TEXT NOT NULL,
    created_on TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
    updated_on TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
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
DROP TABLE IF EXISTS submissions;
