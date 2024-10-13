from asyncpg.connection import Connection
from loguru import logger

from validator.db.database import PSQLDB


async def add_task(
        model_id: str,
        ds_id: str,
        system: str,
        instruction: str,
        input_data: str,
        status: str,
        psql_db: PSQLDB
) -> str:
    async with await psql_db.connection() as connection:
        connection: Connection
        task_id = await connection.fetchval(
            """
            INSERT INTO tasks (model_id, ds_id, system, instruction, input, status)
            VALUES ($1, $2, $3, $4, $5, $6)
            RETURNING task_id
            """,
            model_id,
            ds_id,
            system,
            instruction,
            input_data,
            status,
        )
        return task_id


async def add_node(
        coldkey: str,
        ip: str,
        ip_type: str,
        port: int,
        symmetric_key: str,
        network: float,
        stake: float,
        psql_db: PSQLDB
) -> str:
    async with await psql_db.connection() as connection:
        connection: Connection
        node_id = await connection.fetchval(
            """
            INSERT INTO nodes (coldkey, ip, ip_type, port, symmetric_key, network, stake)
            VALUES ($1, $2, $3, $4, $5, $6, $7)
            RETURNING node_id
            """,
            coldkey,
            ip,
            ip_type,
            port,
            symmetric_key,
            network,
            stake
        )
        return node_id


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
        logger.info(f"Assigned node {node_id} to task {task_id}")


async def add_submission(task_id: str, node_id: str, repo: str, psql_db: PSQLDB) -> str:
    async with await psql_db.connection() as connection:
        connection: Connection
        submission_id = await connection.fetchval(
            """
            INSERT INTO submissions (task_id, node_id, repo)
            VALUES ($1, $2, $3)
            RETURNING submission_id
            """,
            task_id,
            node_id,
            repo
        )
        logger.info(f"submission received: {submission_id} for task: {task_id} from node: {node_id}")
        return submission_id


async def get_task(task_id: str, psql_db: PSQLDB):
    async with await psql_db.connection() as connection:
        connection: Connection
        return await connection.fetchrow(
            """
            SELECT * FROM tasks WHERE task_id = $1
            """,
            task_id,
        )

async def get_tasks_by_status(status: str, psql_db: PSQLDB):
    async with await psql_db.connection() as connection:
        connection: Connection
        return await connection.fetch(
            """
            SELECT * FROM tasks WHERE status = $1
            """,
            status,
        )

async def update_task_columns(task_id: str, updates: dict, psql_db: PSQLDB) -> None:
    async with await psql_db.connection() as connection:
        connection: Connection
        set_clause = ", ".join([f"{column} = ${i+2}" for i, column in enumerate(updates.keys())])
        values = list(updates.values())
        query = f"""
            UPDATE tasks
            SET {set_clause}, updated_timestamp = CURRENT_TIMESTAMP
            WHERE task_id = $1
        """
        await connection.execute(query, task_id, *values)

async def get_node(node_id: str, psql_db: PSQLDB):
    async with await psql_db.connection() as connection:
        connection: Connection
        return await connection.fetchrow(
            """
            SELECT * FROM nodes WHERE node_id = $1
            """,
            node_id,
        )

async def get_all_miners(psql_db: PSQLDB):
    async with await psql_db.connection() as connection:
        connection: Connection
        return await connection.fetch(
            """
            SELECT * FROM nodes
            WHERE trust IS NOT NULL
            """
        )

async def get_all_validators(psql_db: PSQLDB):
    async with await psql_db.connection() as connection:
        connection: Connection
        return await connection.fetch(
            """
            SELECT * FROM nodes
            WHERE vtrust IS NOT NULL
            """
        )

async def get_nodes_assigned_to_task(task_id: str, psql_db: PSQLDB):
    async with await psql_db.connection() as connection:
        connection: Connection
        return await connection.fetch(
            """
            SELECT nodes.* FROM nodes
            JOIN task_nodes ON nodes.node_id = task_nodes.node_id
            WHERE task_nodes.task_id = $1
            """,
            task_id,
        )

async def get_miners_assigned_to_task(task_id: str, psql_db: PSQLDB):
    async with await psql_db.connection() as connection:
        connection: Connection
        return await connection.fetch(
            """
            SELECT nodes.* FROM nodes
            JOIN task_nodes ON nodes.node_id = task_nodes.node_id
            WHERE task_nodes.task_id = $1
            AND nodes.trust IS NOT NULL
            """,
            task_id,
        )


async def get_submission(task_id: str, node_id: str, psql_db: PSQLDB):
    async with await psql_db.connection() as connection:
        connection: Connection
        return await connection.fetchrow(
            """
            SELECT * FROM submissions WHERE task_id = $1 AND node_id = $2
            """,
            task_id,
            node_id,
        )

async def get_miner_latest_submission(task_id: str, node_id: str, psql_db: PSQLDB):
    async with await psql_db.connection() as connection:
        connection: Connection
        return await connection.fetchrow(
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
