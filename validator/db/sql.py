from datetime import datetime
import json
from typing import Dict
from typing import List
from typing import Optional
from uuid import UUID

from asyncpg.connection import Connection

from validator.core.models import Node, TaskNode, TaskResults
from validator.core.models import Submission
from validator.core.models import Task
from validator.db.database import PSQLDB


async def add_task(task: Task, psql_db: PSQLDB) -> Task:
    async with await psql_db.connection() as connection:
        connection: Connection
        task_id = await connection.fetchval(
            """
            INSERT INTO tasks (model_id, ds_id, system, instruction, input, status, hours_to_complete, output, user_id)
            VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9)
            RETURNING task_id
            """,
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


async def get_tasks_with_status(status: str, psql_db: PSQLDB) -> List[Task]:
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
            INSERT INTO nodes (coldkey, ip, ip_type, port, symmetric_key, network, stake, node_id)
            VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
            RETURNING node_id
            """,
            node.coldkey,
            node.ip,
            node.ip_type,
            node.port,
            node.symmetric_key,
            node.network,
            node.stake,
            node.node_id
        )
        return await get_node(node_id, psql_db)


async def get_node(node_id: int, psql_db: PSQLDB) -> Optional[Node]:
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
            submission.repo,
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


async def get_tasks_with_miners_by_user(user_id: str, psql_db: PSQLDB) -> List[Dict]:
    async with await psql_db.connection() as connection:
        connection: Connection
        rows = await connection.fetch(
            """
            SELECT tasks.*, json_agg(
            json_build_object('node_id', nodes.node_id, 'coldkey', nodes.coldkey, 'trust', nodes.trust)
            ) AS miners
            FROM tasks
            LEFT JOIN task_nodes ON tasks.task_id = task_nodes.task_id
            LEFT JOIN nodes ON task_nodes.node_id = nodes.node_id
            WHERE tasks.user_id = $1
            GROUP BY tasks.task_id
            """,
            user_id,
        )
        return [
            {**dict(row), "miners": json.loads(row["miners"]) if isinstance(row["miners"], str) else row["miners"]}
            for row in rows
        ]


async def assign_node_to_task(task_id: str, node_id: int, psql_db: PSQLDB) -> None:
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
    for field, value in updated_task.dict(exclude_unset=True, exclude={"assigned_miners", "updated_timestamp"}).items():
        if getattr(existing_task, field) != value:
            updates[field] = value

    async with await psql_db.connection() as connection:
        connection: Connection
        async with connection.transaction():
            # Update the tasks table
            if updates:
                set_clause = ", ".join([f"{column} = ${i+2}" for i, column in enumerate(updates.keys())])
                values = list(updates.values())
                query = f"""
                    UPDATE tasks
                    SET {set_clause}{', ' if updates else ''}updated_timestamp = CURRENT_TIMESTAMP
                    WHERE task_id = $1
                    RETURNING *
                """
                await connection.execute(query, updated_task.task_id, *values)
            else:
                # If there are no other updates, just update the timestamp
                query = """
                    UPDATE tasks
                    SET updated_timestamp = CURRENT_TIMESTAMP
                    WHERE task_id = $1
                    RETURNING *
                """
                await connection.execute(query, updated_task.task_id)
            # Update the task_nodes table
            if updated_task.assigned_miners is not None:
                # Remove existing assignments
                await connection.execute("DELETE FROM task_nodes WHERE task_id = $1", updated_task.task_id)
                # Add new assignments
                if updated_task.assigned_miners:
                    await connection.executemany(
                        "INSERT INTO task_nodes (task_id, node_id) VALUES ($1, $2)",
                        [(updated_task.task_id, miner_id) for miner_id in updated_task.assigned_miners],
                    )

    return await get_task(updated_task.task_id, psql_db)

# NOTE: Why are we not happy with NULL trust?
async def get_all_miners(psql_db: PSQLDB) -> List[Node]:
    async with await psql_db.connection() as connection:
        connection: Connection
        rows = await connection.fetch(
            # select * :0
            """
            SELECT * FROM nodes
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


async def get_miner_latest_submission(task_id: str, node_id: int, psql_db: PSQLDB) -> Optional[Submission]:
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


async def is_miner_assigned_to_task(task_id: str, node_id: int, psql_db: PSQLDB) -> bool:
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


async def submission_repo_is_unique(repo: str, psql_db: PSQLDB) -> bool:
    async with await psql_db.connection() as connection:
        connection: Connection
        result = await connection.fetchval(
            """
            SELECT 1 FROM submissions
            WHERE repo = $1
            LIMIT 1
            """,
            repo,
        )
        return result is None


async def get_tasks_ready_to_evaluate(psql_db: PSQLDB) -> List[Task]:
    async with await psql_db.connection() as connection:
        connection: Connection
        rows = await connection.fetch(
            """
            SELECT * FROM tasks
            WHERE status = 'training'
            AND NOW() AT TIME ZONE 'UTC' > end_timestamp AT TIME ZONE 'UTC'
            """
        )
        return [Task(**dict(row)) for row in rows]


async def set_task_node_quality_score(task_id: UUID, node_id: int, quality_score: float, psql_db: PSQLDB) -> None:
    async with await psql_db.connection() as connection:
        connection: Connection
        await connection.execute(
            """
            INSERT INTO task_nodes (task_id, node_id, quality_score)
            VALUES ($1, $2, $3)
            ON CONFLICT (task_id, node_id) DO UPDATE
            SET quality_score = $3
            """,
            task_id,
            node_id,
            quality_score,
        )


async def get_task_node_quality_score(task_id: UUID, node_id: int, psql_db: PSQLDB) -> Optional[float]:
    async with await psql_db.connection() as connection:
        connection: Connection
        score = await connection.fetchval(
            """
            SELECT quality_score
            FROM task_nodes
            WHERE task_id = $1 AND node_id = $2
            """,
            task_id,
            node_id,
        )
        return score


async def get_all_quality_scores_for_task(task_id: UUID, psql_db: PSQLDB) -> Dict[UUID, float]:
    async with await psql_db.connection() as connection:
        connection: Connection
        rows = await connection.fetch(
            """
            SELECT node_id, quality_score
            FROM task_nodes
            WHERE task_id = $1 AND quality_score IS NOT NULL
            """,
            task_id,
        )
        return {row["node_id"]: row["quality_score"] for row in rows}


async def set_multiple_task_node_quality_scores(task_id: UUID, quality_scores: Dict[UUID, float], psql_db: PSQLDB) -> None:
    async with await psql_db.connection() as connection:
        connection: Connection
        async with connection.transaction():
            await connection.executemany(
                """
                INSERT INTO task_nodes (task_id, node_id, quality_score)
                VALUES ($1, $2, $3)
                ON CONFLICT (task_id, node_id) DO UPDATE
                SET quality_score = EXCLUDED.quality_score
                """,
                [(task_id, node_id, score) for node_id, score in quality_scores.items()],
            )


async def get_tasks_by_user(user_id: str, psql_db: PSQLDB) -> List[Task]:
    async with await psql_db.connection() as connection:
        rows = await connection.fetch(
            """
            SELECT * FROM tasks WHERE user_id = $1
            """,
            user_id,
        )
        return [Task(**dict(row)) for row in rows]


async def delete_task(task_id: UUID, psql_db: PSQLDB) -> None:
    async with await psql_db.connection() as connection:
        await connection.execute(
            """
            DELETE FROM tasks WHERE task_id = $1
            """,
            task_id,
        )


from datetime import datetime, timedelta
import json

async def get_aggregate_scores_since(start_time: datetime, psql_db) -> list[TaskResults]:
    """
    Get aggregate scores for all completed tasks since the given start time.
    Only includes tasks that have at least one node with score > 0.

    Args:
        start_time: datetime object to filter tasks created after this time
        psql_db: Database connection
    Returns:
        List of TaskResults containing task info and node scores
    """
    async with await psql_db.connection() as connection:
        rows = await connection.fetch(
            """
            SELECT
                t.task_id,
                t.model_id,
                t.ds_id,
                t.system,
                t.instruction,
                t.input,
                t.output,
                t.status,
                t.test_data,
                t.synthetic_data,
                t.hf_training_repo,
                t.miner_scores,
                t.hours_to_complete,
                t.created_timestamp,
                t.updated_timestamp,
                t.started_timestamp,
                t.completed_timestamp,
                COALESCE(
                    json_agg(
                        json_build_object(
                            'task_id', t.task_id::text,
                            'node_id', tn.node_id,
                            'quality_score', tn.quality_score
                        )
                        ORDER BY tn.quality_score DESC NULLS LAST
                    ) FILTER (WHERE tn.node_id IS NOT NULL),
                    '[]'::json
                ) as node_scores
            FROM tasks t
            LEFT JOIN task_nodes tn ON t.task_id = tn.task_id
            WHERE t.created_timestamp >= $1
            AND EXISTS (
                SELECT 1
                FROM task_nodes tn2
                WHERE tn2.task_id = t.task_id
                AND tn2.quality_score > 0
            )
            GROUP BY
                t.task_id,
                t.model_id,
                t.ds_id,
                t.system,
                t.instruction,
                t.input,
                t.output,
                t.status,
                t.test_data,
                t.synthetic_data,
                t.hf_training_repo,
                t.miner_scores,
                t.hours_to_complete,
                t.created_timestamp,
                t.updated_timestamp,
                t.started_timestamp,
                t.completed_timestamp
            ORDER BY t.created_timestamp DESC
            """,
            start_time
        )

        results = []
        for row in rows:
            # Convert row to dict and handle the node_scores separately
            row_dict = dict(row)

            # Create Task object without node_scores
            task_dict = {k: v for k, v in row_dict.items() if k != 'node_scores'}
            task = Task(**task_dict)

            # Parse node_scores - ensure we're working with a list of dicts
            node_scores_data = row_dict['node_scores']
            if isinstance(node_scores_data, str):
                # If it's a string, parse it as JSON
                node_scores_data = json.loads(node_scores_data)

            # Create TaskNode objects
            node_scores = [
                TaskNode(
                    task_id=str(node['task_id']),
                    node_id=int(node['node_id']),
                    quality_score=float(node['quality_score']) if node['quality_score'] is not None else None
                )
                for node in node_scores_data
            ]

            results.append(TaskResults(task=task, node_scores=node_scores))

        return results
