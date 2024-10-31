# submissions.py
from datetime import datetime
import json
from typing import Dict, List, Optional
from uuid import UUID
from asyncpg.connection import Connection

from validator.core.models import TaskNode, TaskResults, Submission, Task
from validator.db.database import PSQLDB
from validator.db.constants import *

async def add_submission(submission: Submission, psql_db: PSQLDB) -> Submission:
    async with await psql_db.connection() as connection:
        connection: Connection
        query = f"""
            INSERT INTO {SUBMISSIONS_TABLE} (
                {TASK_ID}, {HOTKEY}, {NETUID}, {REPO}
            )
            VALUES ($1, $2, $3, $4)
            RETURNING {SUBMISSION_ID}
        """
        submission_id = await connection.fetchval(
            query,
            submission.task_id,
            submission.hotkey,
            submission.netuid,
            submission.repo,
        )
        return await get_submission(submission_id, psql_db)

async def get_submission(submission_id: UUID, psql_db: PSQLDB) -> Optional[Submission]:
    async with await psql_db.connection() as connection:
        connection: Connection
        query = f"""
            SELECT * FROM {SUBMISSIONS_TABLE} WHERE {SUBMISSION_ID} = $1
        """
        row = await connection.fetchrow(query, submission_id)
        if row:
            return Submission(**dict(row))
        return None

async def get_submissions_by_task(task_id: UUID, psql_db: PSQLDB) -> List[Submission]:
    async with await psql_db.connection() as connection:
        connection: Connection
        query = f"""
            SELECT * FROM {SUBMISSIONS_TABLE} WHERE {TASK_ID} = $1
        """
        rows = await connection.fetch(query, task_id)
        return [Submission(**dict(row)) for row in rows]

async def get_node_latest_submission(task_id: str, hotkey: str, netuid: int, psql_db: PSQLDB) -> Optional[Submission]:
    async with await psql_db.connection() as connection:
        connection: Connection
        query = f"""
            SELECT * FROM {SUBMISSIONS_TABLE}
            WHERE {TASK_ID} = $1
            AND {HOTKEY} = $2
            AND {NETUID} = $3
            ORDER BY {CREATED_ON} DESC
            LIMIT 1
        """
        row = await connection.fetchrow(query, task_id, hotkey, netuid)
        if row:
            return Submission(**dict(row))
        return None

async def submission_repo_is_unique(repo: str, psql_db: PSQLDB) -> bool:
    async with await psql_db.connection() as connection:
        connection: Connection
        query = f"""
            SELECT 1 FROM {SUBMISSIONS_TABLE}
            WHERE {REPO} = $1
            LIMIT 1
        """
        result = await connection.fetchval(query, repo)
        return result is None

async def set_task_node_quality_score(task_id: UUID, hotkey: str, netuid: int, quality_score: float, psql_db: PSQLDB) -> None:
    async with await psql_db.connection() as connection:
        connection: Connection
        query = f"""
            INSERT INTO {TASK_NODES_TABLE} (
                {TASK_ID}, {HOTKEY}, {NETUID}, {TASK_NODE_QUALITY_SCORE}
            )
            VALUES ($1, $2, $3, $4)
            ON CONFLICT ({TASK_ID}, {HOTKEY}, {NETUID}) DO UPDATE
            SET {TASK_NODE_QUALITY_SCORE} = $4
        """
        await connection.execute(query, task_id, hotkey, netuid, quality_score)

async def get_task_node_quality_score(task_id: UUID, hotkey: str, netuid: int, psql_db: PSQLDB) -> Optional[float]:
    async with await psql_db.connection() as connection:
        connection: Connection
        query = f"""
            SELECT {TASK_NODE_QUALITY_SCORE}
            FROM {TASK_NODES_TABLE}
            WHERE {TASK_ID} = $1 
            AND {HOTKEY} = $2 
            AND {NETUID} = $3
        """
        return await connection.fetchval(query, task_id, hotkey, netuid)

async def get_all_quality_scores_for_task(task_id: UUID, psql_db: PSQLDB) -> Dict[tuple[str, int], float]:
    async with await psql_db.connection() as connection:
        connection: Connection
        query = f"""
            SELECT {HOTKEY}, {NETUID}, {TASK_NODE_QUALITY_SCORE}
            FROM {TASK_NODES_TABLE}
            WHERE {TASK_ID} = $1 AND {TASK_NODE_QUALITY_SCORE} IS NOT NULL
        """
        rows = await connection.fetch(query, task_id)
        return {(row[HOTKEY], row[NETUID]): row[TASK_NODE_QUALITY_SCORE] for row in rows}

async def set_multiple_task_node_quality_scores(
    task_id: UUID, 
    quality_scores: Dict[tuple[str, int], float], 
    psql_db: PSQLDB
) -> None:
    async with await psql_db.connection() as connection:
        connection: Connection
        async with connection.transaction():
            query = f"""
                INSERT INTO {TASK_NODES_TABLE} (
                    {TASK_ID}, {HOTKEY}, {NETUID}, {TASK_NODE_QUALITY_SCORE}
                )
                VALUES ($1, $2, $3, $4)
                ON CONFLICT ({TASK_ID}, {HOTKEY}, {NETUID}) DO UPDATE
                SET {TASK_NODE_QUALITY_SCORE} = EXCLUDED.{TASK_NODE_QUALITY_SCORE}
            """
            await connection.executemany(
                query,
                [(task_id, node_key[0], node_key[1], score) 
                 for node_key, score in quality_scores.items()]
            )

async def get_aggregate_scores_since(start_time: datetime, psql_db: PSQLDB) -> List[TaskResults]:
    """
    Get aggregate scores for all completed tasks since the given start time.
    Only includes tasks that have at least one node with score > 0.
    """
    async with await psql_db.connection() as connection:
        connection: Connection
        query = f"""
            SELECT
                t.*,
                COALESCE(
                    json_agg(
                        json_build_object(
                            '{TASK_ID}', t.{TASK_ID}::text,
                            '{HOTKEY}', tn.{HOTKEY},
                            '{NETUID}', tn.{NETUID},
                            '{QUALITY_SCORE}', tn.{TASK_NODE_QUALITY_SCORE}
                        )
                        ORDER BY tn.{TASK_NODE_QUALITY_SCORE} DESC NULLS LAST
                    ) FILTER (WHERE tn.{HOTKEY} IS NOT NULL),
                    '[]'::json
                ) as node_scores
            FROM {TASKS_TABLE} t
            LEFT JOIN {TASK_NODES_TABLE} tn ON t.{TASK_ID} = tn.{TASK_ID}
            WHERE t.{STATUS} = 'success'
            AND t.created_timestamp >= $1
            AND EXISTS (
                SELECT 1
                FROM {TASK_NODES_TABLE} tn2
                WHERE tn2.{TASK_ID} = t.{TASK_ID}
                AND tn2.{TASK_NODE_QUALITY_SCORE} > 0
            )
            GROUP BY t.{TASK_ID}
            ORDER BY t.created_timestamp DESC
        """
        rows = await connection.fetch(query, start_time)

        results = []
        for row in rows:
            row_dict = dict(row)
            task_dict = {k: v for k, v in row_dict.items() if k != 'node_scores'}
            task = Task(**task_dict)

            node_scores_data = row_dict['node_scores']
            if isinstance(node_scores_data, str):
                node_scores_data = json.loads(node_scores_data)

            node_scores = [
                TaskNode(
                    task_id=str(node[TASK_ID]),
                    hotkey=node[HOTKEY],
                    netuid=int(node[NETUID]),
                    quality_score=float(node[QUALITY_SCORE]) if node[QUALITY_SCORE] is not None else None
                )
                for node in node_scores_data
            ]

            results.append(TaskResults(task=task, node_scores=node_scores))

        return results