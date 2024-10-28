from typing import List
from typing import Optional

from asyncpg.connection import Connection

from validator.core.models import Node
from validator.db.database import PSQLDB

from validator.db import constants as dcst


async def get_all_nodes(psql_db: PSQLDB) -> List[Node]:
    async with await psql_db.connection() as connection:
        connection: Connection
        rows = await connection.fetch(
            """
            SELECT * FROM nodes
            """
        )
        return [Node(**dict(row)) for row in rows]

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



async def update_our_vali_node_in_db(connection: Connection, ss58_address: str, netuid: int) -> None:
    query = f"""
        UPDATE nodes
        SET is_validator = true
        WHERE {dcst.HOTKEY} = $1 AND {dcst.NETUID} = $2
    """
    await connection.execute(query, ss58_address, netuid)


async def get_vali_ss58_address(psql_db: PSQLDB, netuid: int) -> str | None:
    query = f"""
        SELECT
            {dcst.HOTKEY}
        FROM {dcst.NODES_TABLE}
        WHERE {dcst.OUR_VALIDATOR} = true AND {dcst.NETUID} = $1
    """

    node = await psql_db.fetchone(query, netuid)

    if node is None:
        logger.error(f"I cannot find the validator node for netuid {netuid} in the DB. Maybe control node is still syncing?")
        return None

    return node[dcst.HOTKEY]

