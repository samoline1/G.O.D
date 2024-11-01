# submissions.py
import json
import os
from datetime import datetime
from typing import Dict, List, Optional
from uuid import UUID

from asyncpg.connection import Connection

from validator.core.models import Submission, Task, TaskNode, TaskResults
import validator.db.constants as cst
from validator.db.database import PSQLDB

from core.constants import NETUID


async def add_submission(submission: Submission, psql_db: PSQLDB) -> Submission:
    """Add a new submission for the current NETUID"""
    async with await psql_db.connection() as connection:
        connection: Connection
        query = f"""
            INSERT INTO {cst.SUBMISSIONS_TABLE} (
                {cst.TASK_ID}, {cst.HOTKEY}, {cst.NETUID}, {cst.REPO}
            )
            VALUES ($1, $2, $3, $4)
            RETURNING {cst.SUBMISSION_ID}
        """
        submission_id = await connection.fetchval(
            query,
            submission.task_id,
            submission.hotkey,
            NETUID,
            submission.repo,
        )
        return await get_submission(submission_id, psql_db)


async def get_submission(submission_id: UUID, psql_db: PSQLDB) -> Optional[Submission]:
    """Get a submission by its ID"""
    async with await psql_db.connection() as connection:
        connection: Connection
        query = f"""
            SELECT * FROM {cst.SUBMISSIONS_TABLE} WHERE {cst.SUBMISSION_ID} = $1
        """
        row = await connection.fetchrow(query, submission_id)
        if row:
            return Submission(**dict(row))
        return None


async def get_submissions_by_task(task_id: UUID, psql_db: PSQLDB) -> List[Submission]:
    """Get all submissions for a task"""
    async with await psql_db.connection() as connection:
        connection: Connection
        query = f"""
            SELECT * FROM {cst.SUBMISSIONS_TABLE}
            WHERE {cst.TASK_ID} = $1 AND {cst.NETUID} = $2
        """
        rows = await connection.fetch(query, task_id, NETUID)
        return [Submission(**dict(row)) for row in rows]


async def get_node_latest_submission(task_id: str, hotkey: str, psql_db: PSQLDB) -> Optional[Submission]:
    """Get the latest submission for a node on a task"""
    async with await psql_db.connection() as connection:
        connection: Connection
        query = f"""
            SELECT * FROM {cst.SUBMISSIONS_TABLE}
            WHERE {cst.TASK_ID} = $1
            AND {cst.HOTKEY} = $2
            AND {cst.NETUID} = $3
            ORDER BY {cst.CREATED_ON} DESC
            LIMIT 1
        """
        row = await connection.fetchrow(query, task_id, hotkey, NETUID)
        if row:
            return Submission(**dict(row))
        return None


async def submission_repo_is_unique(repo: str, psql_db: PSQLDB) -> bool:
    """Check if a repository URL is unique"""
    async with await psql_db.connection() as connection:
        connection: Connection
        query = f"""
            SELECT 1 FROM {cst.SUBMISSIONS_TABLE}
            WHERE {cst.REPO} = $1 AND {cst.NETUID} = $2
            LIMIT 1
        """
        result = await connection.fetchval(query, repo, NETUID)
        return result is None


async def set_task_node_quality_score(task_id: UUID, hotkey: str, quality_score: float, psql_db: PSQLDB) -> None:
    """Set quality score for a node's task submission"""
    async with await psql_db.connection() as connection:
        connection: Connection
        query = f"""
            INSERT INTO {cst.TASK_NODES_TABLE} (
                {cst.TASK_ID}, {cst.HOTKEY}, {cst.NETUID}, {cst.TASK_NODE_QUALITY_SCORE}
            )
            VALUES ($1, $2, $3, $4)
            ON CONFLICT ({cst.TASK_ID}, {cst.HOTKEY}, {cst.NETUID}) DO UPDATE
            SET {cst.TASK_NODE_QUALITY_SCORE} = $4
        """
        await connection.execute(query, task_id, hotkey, NETUID, quality_score)


async def get_task_node_quality_score(task_id: UUID, hotkey: str, psql_db: PSQLDB) -> Optional[float]:
    """Get quality score for a node's task submission"""
    async with await psql_db.connection() as connection:
        connection: Connection
        query = f"""
            SELECT {cst.TASK_NODE_QUALITY_SCORE}
            FROM {cst.TASK_NODES_TABLE}
            WHERE {cst.TASK_ID} = $1
            AND {cst.HOTKEY} = $2
            AND {cst.NETUID} = $3
        """
        return await connection.fetchval(query, task_id, hotkey, NETUID)


async def get_all_quality_scores_for_task(task_id: UUID, psql_db: PSQLDB) -> Dict[str, float]:
    """Get all quality scores for a task, keyed by hotkey"""
    async with await psql_db.connection() as connection:
        connection: Connection
        query = f"""
            SELECT {cst.HOTKEY}, {cst.TASK_NODE_QUALITY_SCORE}
            FROM {cst.TASK_NODES_TABLE}
            WHERE {cst.TASK_ID} = $1
            AND {cst.NETUID} = $2
            AND {cst.TASK_NODE_QUALITY_SCORE} IS NOT NULL
        """
        rows = await connection.fetch(query, task_id, NETUID)
        return {row[cst.HOTKEY]: row[cst.TASK_NODE_QUALITY_SCORE] for row in rows}


async def set_multiple_task_node_quality_scores(
    task_id: UUID,
    quality_scores: Dict[str, float],
    psql_db: PSQLDB
) -> None:
    """Set multiple quality scores for task nodes"""
    async with await psql_db.connection() as connection:
        connection: Connection
        async with connection.transaction():
            query = f"""
                INSERT INTO {cst.TASK_NODES_TABLE} (
                    {cst.TASK_ID}, {cst.HOTKEY}, {cst.NETUID}, {cst.TASK_NODE_QUALITY_SCORE}
                )
                VALUES ($1, $2, $3, $4)
                ON CONFLICT ({cst.TASK_ID}, {cst.HOTKEY}, {cst.NETUID}) DO UPDATE
                SET {cst.TASK_NODE_QUALITY_SCORE} = EXCLUDED.{cst.TASK_NODE_QUALITY_SCORE}
            """
            await connection.executemany(
                query,
                [(task_id, hotkey, NETUID, score)
                 for hotkey, score in quality_scores.items()]
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
                            '{cst.TASK_ID}', t.{cst.TASK_ID}::text,
                            '{cst.HOTKEY}', tn.{cst.HOTKEY},
                            '{cst.QUALITY_SCORE}', tn.{cst.TASK_NODE_QUALITY_SCORE}
                        )
                        ORDER BY tn.{cst.TASK_NODE_QUALITY_SCORE} DESC NULLS LAST
                    ) FILTER (WHERE tn.{cst.HOTKEY} IS NOT NULL),
                    '[]'::json
                ) as node_scores
            FROM {cst.TASKS_TABLE} t
            LEFT JOIN {cst.TASK_NODES_TABLE} tn ON t.{cst.TASK_ID} = tn.{cst.TASK_ID}
            WHERE t.{STATUS} = 'success'
            AND t.created_timestamp >= $1
            AND tn.{cst.NETUID} = $2
            AND EXISTS (
                SELECT 1
                FROM {cst.TASK_NODES_TABLE} tn2
                WHERE tn2.{cst.TASK_ID} = t.{cst.TASK_ID}
                AND tn2.{cst.TASK_NODE_QUALITY_SCORE} > 0
                AND tn2.{cst.NETUID} = $2
            )
            GROUP BY t.{cst.TASK_ID}
            ORDER BY t.created_timestamp DESC
        """
        rows = await connection.fetch(query, start_time, NETUID)

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
                    task_id=str(node[cst.TASK_ID]),
                    hotkey=node[cst.HOTKEY],
                    quality_score=float(node[cst.QUALITY_SCORE]) if node[cst.QUALITY_SCORE] is not None else None
                )
                for node in node_scores_data
            ]

            results.append(TaskResults(task=task, node_scores=node_scores))

        return results
