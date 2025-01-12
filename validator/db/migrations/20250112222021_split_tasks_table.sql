-- migrate:up
DO $$
BEGIN
    IF NOT EXISTS (SELECT 1 FROM pg_type WHERE typname = 'tasktype') THEN
        CREATE TYPE tasktype AS ENUM ('TextTask', 'ImageTask');
    END IF;
END $$;

CREATE TABLE IF NOT EXISTS text_tasks (
    task_id UUID PRIMARY KEY,
    ds_id TEXT,
    field_system TEXT,
    field_instruction TEXT NOT NULL,
    field_input TEXT,
    field_output TEXT,
    synthetic_data TEXT,
    format TEXT,
    no_input_format TEXT,
    system_format TEXT
);

CREATE TABLE IF NOT EXISTS image_tasks (
    task_id UUID PRIMARY KEY,
    ds_url TEXT NOT NULL
);

ALTER TABLE tasks ADD COLUMN task_type tasktype;

UPDATE tasks SET task_type = 'TextTask' WHERE task_id IN (SELECT task_id FROM text_tasks);

UPDATE tasks SET task_type = 'ImageTask' WHERE task_id IN (SELECT task_id FROM image_tasks);

ALTER TABLE tasks ALTER COLUMN task_type SET NOT NULL;

INSERT INTO text_tasks (task_id, ds_id, field_system, field_instruction, field_input, field_output, synthetic_data, format, no_input_format, system_format)
SELECT task_id, ds_id, field_system, field_instruction, field_input, field_output, synthetic_data, format, no_input_format, system_format FROM tasks;

ALTER TABLE tasks
DROP COLUMN ds_id,
DROP COLUMN field_system,
DROP COLUMN field_instruction,
DROP COLUMN field_input,
DROP COLUMN field_output,
DROP COLUMN synthetic_data,
DROP COLUMN format,
DROP COLUMN no_input_format,
DROP COLUMN system_format;


-- migrate:down
ALTER TABLE tasks
ADD COLUMN ds_id TEXT,
ADD COLUMN field_system TEXT,
ADD COLUMN field_instruction TEXT NOT NULL,
ADD COLUMN field_input TEXT,
ADD COLUMN field_output TEXT,
ADD COLUMN synthetic_data TEXT,
ADD COLUMN format TEXT,
ADD COLUMN no_input_format TEXT,
ADD COLUMN system_format TEXT;

UPDATE tasks
SET ds_id = t.ds_id,
    field_system = t.field_system,
    field_instruction = t.field_instruction,
    field_input = t.field_input,
    field_output = t.field_output,
    synthetic_data = t.synthetic_data,
    format = t.format,
    no_input_format = t.no_input_format,
    system_format = t.system_format
FROM text_tasks t
WHERE tasks.task_id = t.task_id;

DROP TABLE text_tasks;

DROP TABLE image_tasks;

ALTER TABLE tasks
DROP COLUMN task_type;

DO $$
BEGIN
    IF EXISTS (SELECT 1 FROM pg_type WHERE typname = 'tasktype') THEN
        DROP TYPE tasktype;
    END IF;
END $$;

