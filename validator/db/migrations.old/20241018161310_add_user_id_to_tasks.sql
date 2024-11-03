-- migrate:up
ALTER TABLE tasks
ADD COLUMN user_id TEXT;

-- migrate:down
ALTER TABLE tasks
DROP COLUMN IF EXISTS user_id;
