# tasks.py
import json
from typing import Dict, List, Optional
from uuid import UUID
from asyncpg.connection import Connection

from validator.core.models import Task
from fiber.networking.models import NodeWithFernet as Node
from validator.db.database import PSQLDB
from validator.db.constants import *

async def add_task(task: Task, psql_db: PSQLDB) -> Task:
    async with await psql_db.connection() as connection:
        connection: Connection
        query = f"""
            INSERT INTO {TASKS_TABLE}
            ({MODEL_ID}, {DS_ID}, {SYSTEM}, {INSTRUCTION}, {INPUT}, {STATUS},
             {HOURS_TO_COMPLETE}, {OUTPUT}, {USER_ID})
            VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9)
            RETURNING {TASK_ID}
        """
        task_id = await connection.fetchval(
            query,
            task.model_id,
            task.ds_id,
            task.system,
            task.instruction,
            task.input,
            task.status,
            task.hours_to_complete,
            task.output,
            task.user_id,
        )
        return await get_task(task_id, psql_db)

async def get_nodes_assigned_to_task(task_id: str, psql_db: PSQLDB) -> List[Node]:
    async with await psql_db.connection() as connection:
        connection: Connection
        rows = await connection.fetch(
            """
            SELECT nodes.* FROM nodes
            JOIN task_nodes ON nodes.hotkey = task_nodes.hotkey
            AND nodes.node_id = task_nodes.node_id
            WHERE task_nodes.task_id = $1
            """,
            task_id,
        )
        return [Node(**dict(row)) for row in rows]

async def get_task(task_id: UUID, psql_db: PSQLDB) -> Optional[Task]:
    async with await psql_db.connection() as connection:
        connection: Connection
        query = f"""
            SELECT * FROM {TASKS_TABLE} WHERE {TASK_ID} = $1
        """
        row = await connection.fetchrow(query, task_id)
        if row:
            return Task(**dict(row))
        return None

async def get_tasks_with_status(status: str, psql_db: PSQLDB) -> List[Task]:
    async with await psql_db.connection() as connection:
        connection: Connection
        query = f"""
            SELECT * FROM {TASKS_TABLE} WHERE {STATUS} = $1
        """
        rows = await connection.fetch(query, status)
        return [Task(**dict(row)) for row in rows]

async def get_tasks_with_miners_by_user(user_id: str, psql_db: PSQLDB) -> List[Dict]:
    async with await psql_db.connection() as connection:
        connection: Connection
        query = f"""
            SELECT {TASKS_TABLE}.*, json_agg(
            json_build_object(
                '{NODE_ID}', {NODES_TABLE}.{NODE_ID},
                '{HOTKEY}', {NODES_TABLE}.{HOTKEY},
                '{TRUST}', {NODES_TABLE}.{TRUST}
            )) AS miners
            FROM {TASKS_TABLE}
            LEFT JOIN {TASK_NODES_TABLE} ON {TASKS_TABLE}.{TASK_ID} = {TASK_NODES_TABLE}.{TASK_ID}
            LEFT JOIN {NODES_TABLE} ON {TASK_NODES_TABLE}.{NODE_ID} = {NODES_TABLE}.{NODE_ID}
            WHERE {TASKS_TABLE}.{USER_ID} = $1
            GROUP BY {TASKS_TABLE}.{TASK_ID}
        """
        rows = await connection.fetch(query, user_id)
        return [
            {**dict(row), "miners": json.loads(row["miners"]) if isinstance(row["miners"], str) else row["miners"]}
            for row in rows
        ]


async def assign_node_to_task(task_id: str, node: Node, psql_db: PSQLDB) -> None:
    async with await psql_db.connection() as connection:
        connection: Connection
        query = f"""
            INSERT INTO {TASK_NODES_TABLE} ({TASK_ID}, {HOTKEY}, {NODE_ID})
            VALUES ($1, $2, $3)
        """
        await connection.execute(query, task_id, node.hotkey, node.node_id)

async def update_task(updated_task: Task, psql_db: PSQLDB) -> Task:
    existing_task = await get_task(updated_task.task_id, psql_db)
    if not existing_task:
        raise ValueError("Task not found")

    updates = {}
    for field, value in updated_task.dict(exclude_unset=True, exclude={"assigned_miners", "updated_timestamp"}).items():
        if getattr(existing_task, field) != value:
            updates[field] = value

    async with await psql_db.connection() as connection:
        connection: Connection
        async with connection.transaction():
            if updates:
                set_clause = ", ".join([f"{column} = ${i+2}" for i, column in enumerate(updates.keys())])
                values = list(updates.values())
                query = f"""
                    UPDATE {TASKS_TABLE}
                    SET {set_clause}{', ' if updates else ''}{UPDATED_TIMESTAMP} = CURRENT_TIMESTAMP
                    WHERE {TASK_ID} = $1
                    RETURNING *
                """
                await connection.execute(query, updated_task.task_id, *values)
            else:
                query = f"""
                    UPDATE {TASKS_TABLE}
                    SET {UPDATED_TIMESTAMP} = CURRENT_TIMESTAMP
                    WHERE {TASK_ID} = $1
                    RETURNING *
                """
                await connection.execute(query, updated_task.task_id)

            if updated_task.assigned_miners is not None:
                await connection.execute(
                    f"DELETE FROM {TASK_NODES_TABLE} WHERE {TASK_ID} = $1",
                    updated_task.task_id
                )
                if updated_task.assigned_miners:
                    query = f"""
                        INSERT INTO {TASK_NODES_TABLE} ({TASK_ID}, {HOTKEY}, {NODE_ID})
                        SELECT $1, nodes.{HOTKEY}, nodes.{NODE_ID}
                        FROM {NODES_TABLE} nodes
                        WHERE nodes.{NODE_ID} = ANY($2)
                    """
                    await connection.execute(query, updated_task.task_id, updated_task.assigned_miners)

    return await get_task(updated_task.task_id, psql_db)

async def get_test_set_for_task(task_id: str, psql_db: PSQLDB):
    async with await psql_db.connection() as connection:
        connection: Connection
        query = f"""
            SELECT {TEST_DATA} FROM {TASKS_TABLE}
            WHERE {TASK_ID} = $1
        """
        return await connection.fetchval(query, task_id)

async def get_synthetic_set_for_task(task_id: str, psql_db: PSQLDB):
    async with await psql_db.connection() as connection:
        connection: Connection
        query = f"""
            SELECT {SYNTHETIC_DATA} FROM {TASKS_TABLE}
            WHERE {TASK_ID} = $1
        """
        return await connection.fetchval(query, task_id)

async def get_tasks_ready_to_evaluate(psql_db: PSQLDB) -> List[Task]:
    async with await psql_db.connection() as connection:
        connection: Connection
        query = f"""
            SELECT * FROM {TASKS_TABLE}
            WHERE {STATUS} = 'training'
            AND NOW() AT TIME ZONE 'UTC' > {END_TIMESTAMP} AT TIME ZONE 'UTC'
        """
        rows = await connection.fetch(query)
        return [Task(**dict(row)) for row in rows]

async def get_tasks_by_user(user_id: str, psql_db: PSQLDB) -> List[Task]:
    async with await psql_db.connection() as connection:
        query = f"""
            SELECT * FROM {TASKS_TABLE} WHERE {USER_ID} = $1
        """
        rows = await connection.fetch(query, user_id)
        return [Task(**dict(row)) for row in rows]

async def delete_task(task_id: UUID, psql_db: PSQLDB) -> None:
    async with await psql_db.connection() as connection:
        query = f"""
            DELETE FROM {TASKS_TABLE} WHERE {TASK_ID} = $1
        """
        await connection.execute(query, task_id)
