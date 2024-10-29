from typing import List, Optional
from asyncpg.connection import Connection
from validator.db.database import PSQLDB
from validator.db import constants as dcst
from fiber import utils as futils
from fiber.networking.models import NodeWithFernet as Node
from cryptography.fernet import Fernet
from logging import getLogger

logger = getLogger(__name__)

def create_node_with_fernet(row: dict) -> Optional[Node]:
    """Helper function to create Node object with fernet from database row"""
    try:
        row["fernet"] = Fernet(row[dcst.SYMMETRIC_KEY])
    except Exception as e:
        logger.error(f"Error creating fernet: {e}")
        logger.error(f"node: {row}")
        return None
    return Node(**row)

async def get_all_nodes(psql_db: PSQLDB) -> List[Node]:
    async with await psql_db.connection() as connection:
        connection: Connection
        query = f"""
            SELECT * FROM {dcst.NODES_TABLE}
        """
        rows = await connection.fetch(query)
        nodes = []
        for row in rows:
            node = create_node_with_fernet(dict(row))
            if node:
                nodes.append(node)
        return nodes

async def add_node(node: Node, psql_db: PSQLDB) -> Optional[Node]:
    async with await psql_db.connection() as connection:
        connection: Connection
        query = f"""
            INSERT INTO {dcst.NODES_TABLE} (
                {dcst.NODE_ID}, {dcst.COLDKEY}, {dcst.IP}, {dcst.IP_TYPE},
                {dcst.PORT}, {dcst.SYMMETRIC_KEY}, {dcst.NETWORK}, {dcst.STAKE},
                {dcst.HOTKEY}, {dcst.INCENTIVE}, {dcst.NETUID}, {dcst.LAST_UPDATED},
                {dcst.PROTOCOL}, {dcst.SYMMETRIC_KEY_UUID}, {dcst.OUR_VALIDATOR},
                {dcst.TRUST}, {dcst.VTRUST}
            )
            VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14, $15, $16, $17)
            RETURNING {dcst.HOTKEY}, {dcst.NETUID}
        """
        return_values = await connection.fetchrow(
            query,
            node.node_id,
            node.coldkey,
            node.ip,
            node.ip_type,
            node.port,
            node.symmetric_key,
            node.network,
            node.stake,
            node.hotkey,
            node.incentive,
            node.netuid,
            node.last_updated,
            node.protocol,
            node.symmetric_key_uuid,
            node.our_validator,
            node.trust,
            node.vtrust
        )
        if return_values:
            return await get_node_by_keys(return_values[dcst.HOTKEY], return_values[dcst.NETUID], psql_db)
        return None

async def get_node(node_id: int, psql_db: PSQLDB) -> Optional[Node]:
    async with await psql_db.connection() as connection:
        connection: Connection
        query = f"""
            SELECT * FROM {dcst.NODES_TABLE} WHERE {dcst.NODE_ID} = $1
        """
        row = await connection.fetchrow(query, node_id)
        if row:
            return create_node_with_fernet(dict(row))
        return None

async def get_node_by_keys(hotkey: str, netuid: int, psql_db: PSQLDB) -> Optional[Node]:
    """Get node by primary key (hotkey, netuid)"""
    async with await psql_db.connection() as connection:
        connection: Connection
        query = f"""
            SELECT * FROM {dcst.NODES_TABLE}
            WHERE {dcst.HOTKEY} = $1 AND {dcst.NETUID} = $2
        """
        row = await connection.fetchrow(query, hotkey, netuid)
        if row:
            return create_node_with_fernet(dict(row))
        return None

async def update_our_vali_node_in_db(connection: Connection, ss58_address: str, netuid: int) -> None:
    query = f"""
        UPDATE {dcst.NODES_TABLE}
        SET {dcst.OUR_VALIDATOR} = true
        WHERE {dcst.HOTKEY} = $1 AND {dcst.NETUID} = $2
    """
    await connection.execute(query, ss58_address, netuid)

async def get_vali_ss58_address(psql_db: PSQLDB, netuid: int) -> str | None:
    async with await psql_db.connection() as connection:
        connection: Connection
        query = f"""
            SELECT {dcst.HOTKEY}
            FROM {dcst.NODES_TABLE}
            WHERE {dcst.OUR_VALIDATOR} = true AND {dcst.NETUID} = $1
        """
        row = await connection.fetchrow(query, netuid)
        if row is None:
            logger.error(f"I cannot find the validator node for netuid {netuid} in the DB. Maybe control node is still syncing?")
            return None
        return row[dcst.HOTKEY]

async def insert_symmetric_keys_for_nodes(connection: Connection, nodes: list[Node]) -> None:
    await connection.executemany(
        f"""
        UPDATE {dcst.NODES_TABLE}
        SET {dcst.SYMMETRIC_KEY} = $1, {dcst.SYMMETRIC_KEY_UUID} = $2
        WHERE {dcst.HOTKEY} = $3 and {dcst.NETUID} = $4
        """,
        [
            (futils.fernet_to_symmetric_key(node.fernet), node.symmetric_key_uuid, node.hotkey, node.netuid)
            for node in nodes
            if node.fernet is not None
        ],
    )
