-- migrate:up
ALTER TABLE tasks
ADD COLUMN end_timestamp TIMESTAMPTZ;

-- migrate:down
ALTER TABLE tasks
DROP COLUMN IF EXISTS end_timestamp;
