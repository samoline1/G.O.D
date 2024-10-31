# tasks.py
import json
import os
from typing import Dict, List, Optional
from uuid import UUID

from asyncpg.connection import Connection
from fiber.networking.models import NodeWithFernet as Node

from validator.core.models import Task
from validator.db.constants import *
from validator.db.database import PSQLDB

from core.constants import NETUID


async def add_task(task: Task, psql_db: PSQLDB) -> Task:
    """Add a new task"""
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
    """Get all nodes assigned to a task for the current NETUID"""
    async with await psql_db.connection() as connection:
        connection: Connection
        rows = await connection.fetch(
            f"""
            SELECT nodes.* FROM {NODES_TABLE} nodes
            JOIN {TASK_NODES_TABLE} ON nodes.hotkey = task_nodes.hotkey
            WHERE task_nodes.task_id = $1
            AND nodes.netuid = $2
            AND task_nodes.netuid = $2
            """,
            task_id,
            NETUID
        )
        return [Node(**dict(row)) for row in rows]


async def get_task(task_id: UUID, psql_db: PSQLDB) -> Optional[Task]:
    """Get a task by ID"""
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
    """Get all tasks with a specific status"""
    async with await psql_db.connection() as connection:
        connection: Connection
        query = f"""
            SELECT * FROM {TASKS_TABLE} WHERE {STATUS} = $1
        """
        rows = await connection.fetch(query, status)
        return [Task(**dict(row)) for row in rows]


async def get_tasks_with_miners_by_user(user_id: str, psql_db: PSQLDB) -> List[Dict]:
    """Get all tasks for a user with their assigned miners"""
    async with await psql_db.connection() as connection:
        connection: Connection
        query = f"""
            SELECT {TASKS_TABLE}.*, json_agg(
            json_build_object(
                '{HOTKEY}', {NODES_TABLE}.{HOTKEY},
                '{TRUST}', {NODES_TABLE}.{TRUST}
            )) AS miners
            FROM {TASKS_TABLE}
            LEFT JOIN {TASK_NODES_TABLE} ON {TASKS_TABLE}.{TASK_ID} = {TASK_NODES_TABLE}.{TASK_ID}
            LEFT JOIN {NODES_TABLE} ON
                {TASK_NODES_TABLE}.{HOTKEY} = {NODES_TABLE}.{HOTKEY} AND
                {NODES_TABLE}.{NETUID} = $2
            WHERE {TASKS_TABLE}.{USER_ID} = $1
            GROUP BY {TASKS_TABLE}.{TASK_ID}
        """
        rows = await connection.fetch(query, user_id, NETUID)
        return [
            {**dict(row), "miners": json.loads(row["miners"]) if isinstance(row["miners"], str) else row["miners"]}
            for row in rows
        ]


async def assign_node_to_task(task_id: str, node: Node, psql_db: PSQLDB) -> None:
    """Assign a node to a task"""
    async with await psql_db.connection() as connection:
        connection: Connection
        query = f"""
            INSERT INTO {TASK_NODES_TABLE} ({TASK_ID}, {HOTKEY}, {NETUID})
            VALUES ($1, $2, $3)
        """
        await connection.execute(query, task_id, node.hotkey, NETUID)


async def update_task(updated_task: Task, psql_db: PSQLDB) -> Task:
    """Update a task and its assigned miners"""
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
                    f"DELETE FROM {TASK_NODES_TABLE} WHERE {TASK_ID} = $1 AND {NETUID} = $2",
                    updated_task.task_id,
                    NETUID
                )
                if updated_task.assigned_miners:
                    # Now assuming assigned_miners is just a list of hotkeys
                    query = f"""
                        INSERT INTO {TASK_NODES_TABLE} ({TASK_ID}, {HOTKEY}, {NETUID})
                        SELECT $1, nodes.{HOTKEY}, $3
                        FROM {NODES_TABLE} nodes
                        WHERE nodes.{HOTKEY} = ANY($2)
                        AND nodes.{NETUID} = $3
                    """
                    await connection.execute(query, updated_task.task_id, updated_task.assigned_miners, NETUID)

    return await get_task(updated_task.task_id, psql_db)


async def get_test_set_for_task(task_id: str, psql_db: PSQLDB):
    """Get test data for a task"""
    async with await psql_db.connection() as connection:
        connection: Connection
        query = f"""
            SELECT {TEST_DATA} FROM {TASKS_TABLE}
            WHERE {TASK_ID} = $1
        """
        return await connection.fetchval(query, task_id)


async def get_synthetic_set_for_task(task_id: str, psql_db: PSQLDB):
    """Get synthetic data for a task"""
    async with await psql_db.connection() as connection:
        connection: Connection
        query = f"""
            SELECT {SYNTHETIC_DATA} FROM {TASKS_TABLE}
            WHERE {TASK_ID} = $1
        """
        return await connection.fetchval(query, task_id)


async def get_tasks_ready_to_evaluate(psql_db: PSQLDB) -> List[Task]:
    """Get all tasks ready for evaluation"""
    async with await psql_db.connection() as connection:
        connection: Connection
        query = f"""
            SELECT t.* FROM {TASKS_TABLE} t
            WHERE t.{STATUS} = 'training'
            AND NOW() AT TIME ZONE 'UTC' > t.{END_TIMESTAMP} AT TIME ZONE 'UTC'
            AND EXISTS (
                SELECT 1 FROM {TASK_NODES_TABLE} tn
                WHERE tn.{TASK_ID} = t.{TASK_ID}
                AND tn.{NETUID} = $1
            )
        """
        rows = await connection.fetch(query, NETUID)
        return [Task(**dict(row)) for row in rows]


async def get_tasks_by_user(user_id: str, psql_db: PSQLDB) -> List[Task]:
    """Get all tasks for a user"""
    async with await psql_db.connection() as connection:
        query = f"""
            SELECT DISTINCT t.* FROM {TASKS_TABLE} t
            LEFT JOIN {TASK_NODES_TABLE} tn ON t.{TASK_ID} = tn.{TASK_ID}
            WHERE t.{USER_ID} = $1
            AND (tn.{NETUID} = $2 OR tn.{NETUID} IS NULL)
        """
        rows = await connection.fetch(query, user_id, NETUID)
        return [Task(**dict(row)) for row in rows]


async def delete_task(task_id: UUID, psql_db: PSQLDB) -> None:
    """Delete a task and its associated node assignments"""
    async with await psql_db.connection() as connection:
        async with connection.transaction():
            # First delete task_nodes entries for this netuid
            await connection.execute(
                f"""
                DELETE FROM {TASK_NODES_TABLE}
                WHERE {TASK_ID} = $1 AND {NETUID} = $2
                """,
                task_id,
                NETUID
            )

            # Then delete the task if it has no more node assignments
            await connection.execute(
                f"""
                DELETE FROM {TASKS_TABLE}
                WHERE {TASK_ID} = $1
                AND NOT EXISTS (
                    SELECT 1 FROM {TASK_NODES_TABLE}
                    WHERE {TASK_ID} = $1
                )
                """,
                task_id
            )
