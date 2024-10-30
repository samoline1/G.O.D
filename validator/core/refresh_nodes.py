"""
Gets the latest nodes from the network and stores them in the database,
migrating the old nodes to history in the process
"""

import asyncio
import traceback

import httpx

from fiber.networking.models import NodeWithFernet as Node
from validator.db.sql.nodes import get_all_nodes, add_node, get_last_updated_time_for_nodes, insert_symmetric_keys_for_nodes
from fiber.logging_utils import get_logger
from fiber.chain import fetch_nodes
from validator.core.config import Config
from fiber.validator import handshake, client
from datetime import datetime, timedelta
from cryptography.fernet import Fernet

logger = get_logger(__name__)

def _format_exception(e: Exception) -> str:
    """Format an exception with its traceback for logging."""
    return f"Exception Type: {type(e).__name__}\nException Message: {str(e)}\nTraceback:\n{''.join(traceback.format_tb(e.__traceback__))}"


async def get_and_store_nodes(config: Config) -> list[Node]:
    async with await config.psql_db.connection() as connection:
        if await is_recent_update(connection, config.netuid):
            # come back and add netuid
            #return await get_all_nodes(config.psql_db, config.netuid)
            return await get_all_nodes(config.psql_db)

    raw_nodes = await fetch_nodes_from_substrate(config)

    # Ensuring the Nodes get converted to NodesWithFernet
    nodes = [Node(**node.model_dump(mode="json")) for node in raw_nodes]

    nodes = await store_nodes(config, nodes)
#    await update_our_validator_node(config) debug add back in

    await perform_handshakes(nodes, config)
    logger.info(f"Stored {len(nodes)} nodes.")
    return nodes


async def is_recent_update(connection, netuid: int) -> bool:
    last_updated_time = await get_last_updated_time_for_nodes(connection, netuid)
    if last_updated_time is not None and datetime.now() - last_updated_time.replace(tzinfo=None) < timedelta(minutes=30):
        logger.info(
            f"Last update for nodes table was at {last_updated_time}, which is less than 30 minutes ago - skipping refresh"
        )
        return True
    return False


async def fetch_nodes_from_substrate(config: Config) -> list[Node]:
    # NOTE: Will this cause issues if this method closes the conenction
    # on substrate interface, but we use the same substrate interface object elsewhere?
    return await asyncio.to_thread(fetch_nodes.get_nodes_for_netuid, config.substrate, config.netuid)


async def store_nodes(config: Config, nodes: list[Node]):
    nodef = []
    for node in nodes:
        logger.info(f"Here is a node {node}")
        if node.node_id == 60:
            logger.info(f"Adding in this node {node}")
            nodef.append(node)
    await asyncio.gather(*(add_node(node, config.psql_db) for node in nodef))
    return nodef


async def update_our_validator_node(config: Config):
    async with await config.psql_db.connection() as connection:
        await update_our_vali_node_in_db(connection, config.keypair.ss58_address, config.netuid)


async def _handshake(config: Config, node: Node, async_client: httpx.AsyncClient) -> Node:
    logger.info(f"I am attempting to shake hands with {node}")
    logger.info(f"Here is my config keypair {config.keypair}")
    node_copy = node.model_copy()
    server_address = client.construct_server_address(
        node=node,
        replace_with_docker_localhost=False,
        replace_with_localhost=False,
    )

    try:
        symmetric_key, symmetric_key_uid = await handshake.perform_handshake(
            async_client, server_address, config.keypair, node.hotkey
        )
    except Exception as e:
        error_details = _format_exception(e)
        logger.debug(f"Failed to perform handshake with {server_address}. Details:\n{error_details}")

        if isinstance(e, (httpx.HTTPStatusError, httpx.RequestError, httpx.ConnectError)):
            if hasattr(e, "response"):
                logger.debug(f"Response content: {e.response.text}")

        return node_copy

    fernet = Fernet(symmetric_key)
    node_copy.fernet = fernet
    node_copy.symmetric_key_uuid = symmetric_key_uid
    return node_copy


async def perform_handshakes(nodes: list[Node], config: Config) -> list[Node]:
    tasks = []
    shaked_nodes: list[Node] = []
    for node in nodes:
        if node.fernet is None or node.symmetric_key_uuid is None:
            tasks.append(_handshake(config, node, config.httpx_client))
        if len(tasks) > 50:
            shaked_nodes.extend(await asyncio.gather(*tasks))
            tasks = []

    if tasks:
        shaked_nodes.extend(await asyncio.gather(*tasks))

    nodes_where_handshake_worked = [
        node for node in shaked_nodes if node.fernet is not None and node.symmetric_key_uuid is not None
    ]
    if len(nodes_where_handshake_worked) == 0:
        logger.info("❌ Failed to perform handshakes with any nodes!")
        return []
    logger.info(f"✅ performed handshakes successfully with {len(nodes_where_handshake_worked)} nodes!")

    async with await config.psql_db.connection() as connection:
        await insert_symmetric_keys_for_nodes(connection, nodes_where_handshake_worked)

    return shaked_nodes
