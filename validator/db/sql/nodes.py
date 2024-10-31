import datetime
from logging import getLogger
from typing import List, Optional

from asyncpg.connection import Connection
from fiber import utils as futils
from fiber.networking.models import NodeWithFernet as Node

from validator.utils.query_substrate import query_substrate
from validator.db import constants as dcst
from validator.db.database import PSQLDB
from core.constants import NETUID

logger = getLogger(__name__)


async def get_all_nodes(psql_db: PSQLDB) -> List[Node]:
    """Get all nodes for the current NETUID"""
    async with await psql_db.connection() as connection:
        connection: Connection
        query = f"""
            SELECT * FROM {dcst.NODES_TABLE}
            WHERE {dcst.NETUID} = $1
        """
        rows = await connection.fetch(query, NETUID)
        return [Node(**dict(row)) for row in rows]


async def add_node(node: Node, psql_db: PSQLDB) -> Optional[Node]:
    """Add a new node with the current NETUID"""
    async with await psql_db.connection() as connection:
        connection: Connection
        query = f"""
            INSERT INTO {dcst.NODES_TABLE} (
                {dcst.HOTKEY}, {dcst.NETUID}, {dcst.COLDKEY}, {dcst.IP},
                {dcst.IP_TYPE}, {dcst.PORT}, {dcst.SYMMETRIC_KEY}, {dcst.NETWORK},
                {dcst.STAKE}, {dcst.INCENTIVE}, {dcst.LAST_UPDATED},
                {dcst.PROTOCOL}, {dcst.SYMMETRIC_KEY_UUID}, {dcst.OUR_VALIDATOR},
                {dcst.TRUST}, {dcst.VTRUST}
            )
            VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14, $15, $16)
            RETURNING {dcst.HOTKEY}
        """
        return_value = await connection.fetchval(
            query,
            node.hotkey,
            NETUID,
            node.coldkey,
            node.ip,
            node.ip_type,
            node.port,
            None,
            NETUID,
            node.stake,
            node.incentive,
            node.last_updated,
            node.protocol,
            node.symmetric_key_uuid,
            False,  # assume not our validator
            node.trust,
            node.vtrust
        )
        if return_value:
            return await get_node_by_hotkey(return_value, psql_db)
        return None


async def get_node_by_hotkey(hotkey: str, psql_db: PSQLDB) -> Optional[Node]:
    """Get node by hotkey for the current NETUID"""
    async with await psql_db.connection() as connection:
        connection: Connection
        query = f"""
            SELECT * FROM {dcst.NODES_TABLE}
            WHERE {dcst.HOTKEY} = $1 AND {dcst.NETUID} = $2
        """
        row = await connection.fetchrow(query, hotkey, NETUID)
        if row:
            return Node(**dict(row))
        return None


async def update_our_vali_node_in_db(connection: Connection, ss58_address: str) -> None:
    """Update validator node for the current NETUID"""
    query = f"""
        UPDATE {dcst.NODES_TABLE}
        SET {dcst.OUR_VALIDATOR} = true
        WHERE {dcst.HOTKEY} = $1 AND {dcst.NETUID} = $2
    """
    await connection.execute(query, ss58_address, NETUID)


async def get_vali_ss58_address(psql_db: PSQLDB) -> str | None:
    """Get validator SS58 address for the current NETUID"""
    async with await psql_db.connection() as connection:
        connection: Connection
        query = f"""
            SELECT {dcst.HOTKEY}
            FROM {dcst.NODES_TABLE}
            WHERE {dcst.OUR_VALIDATOR} = true AND {dcst.NETUID} = $1
        """
        row = await connection.fetchrow(query, NETUID)
        if row is None:
            logger.error(f"Cannot find validator node for netuid {NETUID} in the DB. Maybe control node is still syncing?")
            return None
        return row[dcst.HOTKEY]


async def insert_symmetric_keys_for_nodes(connection: Connection, nodes: list[Node]) -> None:
    """Insert symmetric keys for nodes in the current NETUID"""
    await connection.executemany(
        f"""
        UPDATE {dcst.NODES_TABLE}
        SET {dcst.SYMMETRIC_KEY} = $1, {dcst.SYMMETRIC_KEY_UUID} = $2
        WHERE {dcst.HOTKEY} = $3 AND {dcst.NETUID} = $4
        """,
        [
            (futils.fernet_to_symmetric_key(node.fernet), node.symmetric_key_uuid, node.hotkey, NETUID)
            for node in nodes
            if node.fernet is not None
        ],
    )


async def get_last_updated_time_for_nodes(connection: Connection) -> datetime.datetime | None:
    """Get last updated time for nodes in the current NETUID"""
    query = f"""
        SELECT MAX({dcst.CREATED_TIMESTAMP})
        FROM {dcst.NODES_TABLE}
        WHERE {dcst.NETUID} = $1
    """
    return await connection.fetchval(query, NETUID)


async def migrate_nodes_to_history(psql_db: PSQLDB) -> None:
    """Migrate nodes to history table for the current NETUID"""
    logger.debug(f"Migrating nodes to history for NETUID {NETUID}")
    async with await psql_db.connection() as connection:
        await connection.execute(
            f"""
            INSERT INTO {dcst.NODES_HISTORY_TABLE} (
                {dcst.HOTKEY},
                {dcst.COLDKEY},
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
            WHERE {dcst.NETUID} = $1
        """,
            NETUID
        )

        logger.debug(f"Truncating node info table for NETUID {NETUID}")
        await connection.execute(
            f"DELETE FROM {dcst.NODES_TABLE} WHERE {dcst.NETUID} = $1",
            NETUID
        )

async def get_vali_node_id(substrate: SubstrateInterface, netuid: int, ss58_address: str) -> str | None:
    _, uid = query_substrate(
        substrate, "SubtensorModule", "Uids", [netuid, ss58_address], return_value=True
    )
    return uid
