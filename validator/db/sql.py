from typing import List
from typing import Optional
from uuid import UUID

from asyncpg.connection import Connection

from validator.core.models import Node
from validator.core.models import Submission
from validator.core.models import Task
from validator.db.database import PSQLDB


async def add_task(task: Task, psql_db: PSQLDB) -> Task:
    async with await psql_db.connection() as connection:
        connection: Connection
        task_id = await connection.fetchval(
            """
            INSERT INTO tasks (model_id, ds_id, system, instruction, input, status)
            VALUES ($1, $2, $3, $4, $5, $6)
            RETURNING task_id
            """,
            task.model_id,
            task.ds_id,
            task.system,
            task.instruction,
            task.input,
            task.status
        )

        return await get_task(task_id, psql_db)

async def get_task(task_id: UUID, psql_db: PSQLDB) -> Optional[Task]:
    async with await psql_db.connection() as connection:
        connection: Connection
        row = await connection.fetchrow(
            """
            SELECT * FROM tasks WHERE task_id = $1
            """,
            task_id,
        )
        if row:
            return Task(**dict(row))
        return None

async def get_tasks_by_status(status: str, psql_db: PSQLDB) -> List[Task]:
    async with await psql_db.connection() as connection:
        connection: Connection
        rows = await connection.fetch(
            """
            SELECT * FROM tasks WHERE status = $1
            """,
            status,
        )
        return [Task(**dict(row)) for row in rows]

async def add_node(node: Node, psql_db: PSQLDB) -> Node:
    async with await psql_db.connection() as connection:
        connection: Connection
        node_id = await connection.fetchval(
            """
            INSERT INTO nodes (coldkey, ip, ip_type, port, symmetric_key, network, stake)
            VALUES ($1, $2, $3, $4, $5, $6, $7)
            RETURNING node_id
            """,
            node.coldkey,
            node.ip,
            node.ip_type,
            node.port,
            node.symmetric_key,
            node.network,
            node.stake
        )
        return await get_node(node_id, psql_db)

async def get_node(node_id: UUID, psql_db: PSQLDB) -> Optional[Node]:
    async with await psql_db.connection() as connection:
        connection: Connection
        row = await connection.fetchrow(
            """
            SELECT * FROM nodes WHERE node_id = $1
            """,
            node_id,
        )
        if row:
            return Node(**dict(row))
        return None

async def add_submission(submission: Submission, psql_db: PSQLDB) -> Submission:
    async with await psql_db.connection() as connection:
        connection: Connection
        submission_id = await connection.fetchval(
            """
            INSERT INTO submissions (task_id, node_id, repo)
            VALUES ($1, $2, $3)
            RETURNING submission_id
            """,
            submission.task_id,
            submission.node_id,
            submission.repo
        )
        return await get_submission(submission_id, psql_db)

async def get_submission(submission_id: UUID, psql_db: PSQLDB) -> Optional[Submission]:
    async with await psql_db.connection() as connection:
        connection: Connection
        row = await connection.fetchrow(
            """
            SELECT * FROM submissions WHERE submission_id = $1
            """,
            submission_id,
        )
        if row:
            return Submission(**dict(row))
        return None


async def assign_node_to_task(task_id: str, node_id: str, psql_db: PSQLDB) -> None:
    async with await psql_db.connection() as connection:
        connection: Connection
        await connection.execute(
            """
            INSERT INTO task_nodes (task_id, node_id)
            VALUES ($1, $2)
            """,
            task_id,
            node_id,
        )


async def update_task(updated_task: Task, psql_db: PSQLDB) -> Task:
    existing_task = await get_task(updated_task.task_id, psql_db)
    if not existing_task:
        raise ValueError("Task not found")

    updates = {}
    for field, value in updated_task.dict(exclude_unset=True).items():
        if getattr(existing_task, field) != value:
            updates[field] = value

    if not updates:
        return existing_task

    set_clause = ", ".join([f"{column} = ${i+2}" for i, column in enumerate(updates.keys())])
    values = list(updates.values())
    query = f"""
        UPDATE tasks
        SET {set_clause}, updated_timestamp = CURRENT_TIMESTAMP
        WHERE task_id = $1
    """

    async with await psql_db.connection() as connection:
        connection: Connection
        await connection.execute(query, updated_task.task_id, *values)

    return await get_task(updated_task.task_id, psql_db)

async def get_all_miners(psql_db: PSQLDB) -> List[Node]:
    async with await psql_db.connection() as connection:
        connection: Connection
        rows = await connection.fetch(
            """
            SELECT * FROM nodes
            WHERE trust IS NOT NULL
            """
        )
        return [Node(**dict(row)) for row in rows]

async def get_all_validators(psql_db: PSQLDB) -> List[Node]:
    async with await psql_db.connection() as connection:
        connection: Connection
        rows = await connection.fetch(
            """
            SELECT * FROM nodes
            WHERE vtrust IS NOT NULL
            """
        )
        return [Node(**dict(row)) for row in rows]

async def get_nodes_assigned_to_task(task_id: str, psql_db: PSQLDB) -> List[Node]:
    async with await psql_db.connection() as connection:
        connection: Connection
        rows = await connection.fetch(
            """
            SELECT nodes.* FROM nodes
            JOIN task_nodes ON nodes.node_id = task_nodes.node_id
            WHERE task_nodes.task_id = $1
            """,
            task_id,
        )
        return [Node(**dict(row)) for row in rows]


async def get_miners_assigned_to_task(task_id: str, psql_db: PSQLDB) -> List[Node]:
    async with await psql_db.connection() as connection:
        connection: Connection
        rows = await connection.fetch(
            """
            SELECT nodes.* FROM nodes
            JOIN task_nodes ON nodes.node_id = task_nodes.node_id
            WHERE task_nodes.task_id = $1
            AND nodes.trust IS NOT NULL
            """,
            task_id,
        )
        return [Node(**dict(row)) for row in rows]

async def get_submissions_by_task(task_id: UUID, psql_db: PSQLDB) -> List[Submission]:
    async with await psql_db.connection() as connection:
        connection: Connection
        rows = await connection.fetch(
            """
            SELECT * FROM submissions WHERE task_id = $1
            """,
            task_id,
        )
        return [Submission(**dict(row)) for row in rows]

async def get_miner_latest_submission(task_id: str, node_id: str, psql_db: PSQLDB) -> Optional[Submission]:
    async with await psql_db.connection() as connection:
        connection: Connection
        row = await connection.fetchrow(
            """
            SELECT * FROM submissions
            WHERE task_id = $1
            AND node_id = $2
            ORDER BY created_on DESC
            LIMIT 1
            """,
            task_id,
            node_id,
        )
        if row:
            return Submission(**dict(row))
        return None

async def is_miner_assigned_to_task(task_id: str, node_id: str, psql_db: PSQLDB) -> bool:
    async with await psql_db.connection() as connection:
        connection: Connection
        result = await connection.fetchval(
            """
            SELECT 1 FROM task_nodes
            WHERE task_id = $1
            AND node_id = $2
            LIMIT 1
            """,
            task_id,
            node_id,
        )
        return result is not None

async def get_test_set_for_task(task_id: str, psql_db: PSQLDB):
    async with await psql_db.connection() as connection:
        connection: Connection
        return await connection.fetchval(
            """
            SELECT test_data FROM tasks
            WHERE task_id = $1
            """,
            task_id,
        )

async def get_synthetic_set_for_task(task_id: str, psql_db: PSQLDB):
    async with await psql_db.connection() as connection:
        connection: Connection
        return await connection.fetchval(
            """
            SELECT synthetic_data FROM tasks
            WHERE task_id = $1
            """,
            task_id,
        )
