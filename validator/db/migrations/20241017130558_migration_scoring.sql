-- migrate:up
ALTER TABLE task_nodes
ADD COLUMN quality_score FLOAT;

-- migrate:down
ALTER TABLE task_nodes
DROP COLUMN quality_score;
