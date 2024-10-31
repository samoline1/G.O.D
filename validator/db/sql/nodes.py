from typing import List, Optional
from asyncpg.connection import Connection
from fiber.chain.metagraph import SubstrateInterface
from sqlalchemy.sql.operators import ne
from validator.db.database import PSQLDB
from validator.db import constants as dcst
from fiber import utils as futils
from fiber.networking.models import NodeWithFernet as Node
from cryptography.fernet import Fernet
from logging import getLogger
import datetime

from validator.utils.query_substrate import query_substrate

logger = getLogger(__name__)

def create_node_with_fernet(row: dict) -> Optional[Node]:
    """Helper function to create Node object with fernet from database row"""
    try:
        if dcst.SYMMETRIC_KEY in row:
            row["fernet"] = Fernet(row[dcst.SYMMETRIC_KEY])
        else:
            row["fernet"] = None
    except Exception as e:
        logger.error(f"Error creating fernet: {e}")
        logger.error(f"node: {row}")
        return None
    return Node(**row)

async def get_all_nodes(psql_db: PSQLDB, netuid: int) -> List[Node]:
    async with await psql_db.connection() as connection:
        connection: Connection
        query = f"""
            SELECT * FROM {dcst.NODES_TABLE} WHERE {dcst.NETUID} = $1
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

async def get_last_updated_time_for_nodes(connection: Connection, netuid: int) -> datetime.datetime | None:
    query = f"""
        SELECT MAX({dcst.CREATED_TIMESTAMP})
        FROM {dcst.NODES_TABLE}
        WHERE {dcst.NETUID} = $1
    """
    return await connection.fetchval(query, netuid)



async def insert_symmetric_keys_for_nodes(connection: Connection, nodes: list[Node]) -> None:
    logger.info(f"Inserting {len([node for node in nodes if node.fernet is not None])} nodes into {dcst.NODES_TABLE}...")
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

async def get_vali_node_id(substrate: SubstrateInterface, netuid: int, ss58_address: str) -> str | None:
    _, uid = query_substrate(
        substrate, "SubtensorModule", "Uids", [netuid, ss58_address], return_value=True
    )
    return uid


async def migrate_nodes_to_history(connection: Connection) -> None:  # noqa: F821
    logger.debug("Migrating NODEs to NODE history")
    await connection.execute(
        f"""
        INSERT INTO {dcst.NODES_HISTORY_TABLE} (
            {dcst.HOTKEY},
            {dcst.COLDKEY},
            {dcst.NODE_ID},
            {dcst.INCENTIVE},
            {dcst.NETUID},
            {dcst.STAKE},
            {dcst.TRUST},
            {dcst.VTRUST},
            {dcst.LAST_UPDATED},
            {dcst.IP},
            {dcst.IP_TYPE},
            {dcst.PORT},
            {dcst.PROTOCOL},
            {dcst.NETWORK}
        )
        SELECT
            {dcst.HOTKEY},
            {dcst.COLDKEY},
            {dcst.NODE_ID},
            {dcst.INCENTIVE},
            {dcst.NETUID},
            {dcst.STAKE},
            {dcst.TRUST},
            {dcst.VTRUST},
            {dcst.LAST_UPDATED},
            {dcst.IP},
            {dcst.IP_TYPE},
            {dcst.PORT},
            {dcst.PROTOCOL},
            {dcst.NETWORK}
        FROM {dcst.NODES_TABLE}
    """
    )

    logger.debug("Truncating NODE info table")
    await connection.execute(f"DELETE FROM {dcst.NODES_TABLE}")

