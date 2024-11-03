-- migrate:up
CREATE TABLE IF NOT EXISTS tasks (
    task_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    model_id TEXT NOT NULL,
    ds_id TEXT NOT NULL,
    system TEXT,
    instruction TEXT,
    input TEXT NOT NULL,
    output TEXT,
    status TEXT NOT NULL,
    test_data TEXT,
    synthetic_data TEXT,
    hf_training_repo TEXT,
    miner_scores FLOAT[],
    hours_to_complete INTEGER,
    created_timestamp TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
    updated_timestamp TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
    started_timestamp TIMESTAMPTZ,
    completed_timestamp TIMESTAMPTZ
);

-- migrate:down
DROP TABLE IF EXISTS tasks;
