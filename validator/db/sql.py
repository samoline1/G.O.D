from asyncpg.connection import Connection
from loguru import logger

from validator.core import constants
from validator.db.database import PSQLDB


async def store_signing_message(account_id: str, message: str, ss58_address: str, psql_db: PSQLDB) -> None:
    async with await psql_db.connection() as connection:
        connection: Connection

        existing_message = await connection.fetchval(
            """
            SELECT id FROM signing_messages
            WHERE coldkey = $1 AND account_id = $2
            """,
            ss58_address,
            account_id,
        )

        if existing_message:
            await connection.execute(
                """
                UPDATE signing_messages
                SET message = $1, created_at = CURRENT_TIMESTAMP, expired_at = CURRENT_TIMESTAMP + INTERVAL '2 hour'
                WHERE id = $2
                """,
                message,
                existing_message,
            )
            logger.info(f"Updated signing message for coldkey {ss58_address}")
        else:
            await connection.execute(
                """
                INSERT INTO signing_messages (account_id, message, coldkey)
                VALUES ($1, $2, $3)
                """,
                account_id,
                message,
                ss58_address,
            )
            logger.info(f"Inserted new signing message for coldkey {ss58_address}")


async def get_signing_message_and_account_id(message: str, coldkey: str, psql_db: PSQLDB) -> str | None:
    async with await psql_db.connection() as connection:
        connection: Connection
        verified_account_id = await connection.fetchval(
            "SELECT account_id FROM signing_messages WHERE message = $1 and coldkey = $2 and verified = true",
            message,
            coldkey,
        )
        if verified_account_id:
            await connection.execute(
                "UPDATE accounts SET coldkey = $1 WHERE account_id = $2", coldkey, verified_account_id
            )
            return constants.VERIFIED

        return await connection.fetchval(
            "SELECT account_id FROM signing_messages WHERE message = $1 and expired_at > now() and coldkey = $2",
            message,
            coldkey,
        )


async def add_account(account_id: str, coldkey: str, psql_db: PSQLDB) -> None:
    async with await psql_db.connection() as connection:
        connection: Connection

        existing_row = await connection.fetchrow(
            """
            SELECT * FROM accounts WHERE coldkey = $1
            """,
            coldkey,
        )

        if existing_row:
            logger.info(f"Account for coldkey {coldkey} already exists. No action taken.")
        else:
            # Insert if the coldkey does not exist
            await connection.execute(
                """
                INSERT INTO accounts (coldkey, account_id)
                VALUES ($1, $2)
                """,
                coldkey,
                account_id,
            )
            logger.info(f"Added account for coldkey {coldkey}")


async def store_api_key(account_id: str, api_key: str, psql_db: PSQLDB) -> None:
    async with await psql_db.connection() as connection:
        connection: Connection
        await connection.execute(
            "INSERT INTO api_keys (account_id, key) VALUES ($1, $2)",
            account_id,
            api_key,
        )


async def update_api_key_rate_limit(account_id: str, rate_limit_per_minute: int, psql_db: PSQLDB) -> None:
    async with await psql_db.connection() as connection:
        connection: Connection
        await connection.execute(
            "UPDATE api_keys SET rate_limit_per_minute = $1 WHERE account_id = $2",
            rate_limit_per_minute,
            account_id,
        )


async def query_accounts_and_coldkeys(psql_db: PSQLDB) -> list[tuple[str, str]]:
    async with await psql_db.connection() as connection:
        connection: Connection
        return await connection.fetch("SELECT account_id, coldkey FROM accounts WHERE deleted_at IS NULL")


async def get_keys_and_rate_limits(psql_db: PSQLDB, account_id: str) -> list[tuple[str, int]]:
    async with await psql_db.connection() as connection:
        connection: Connection
        return await connection.fetch(
            (
                "SELECT key, validator_address, rate_limit_per_minute FROM api_keys "
                "where expired_at is null and account_id = $1"
            ),
            account_id,
        )


async def add_account_without_coldkey(account_id: str, psql_db: PSQLDB) -> None:
    async with await psql_db.connection() as connection:
        connection: Connection

        await connection.execute(
            """
            INSERT INTO accounts (account_id)
            VALUES ($1)
            """,
            account_id,
        )
        logger.info(f"Added non-wallet account with account_id {account_id}")


async def get_coldkey_by_api_key(api_key: str, psql_db) -> str | None:
    async with await psql_db.connection() as connection:
        query = """
        SELECT accounts.coldkey
        FROM api_keys
        JOIN accounts ON api_keys.account_id = accounts.account_id
        WHERE api_keys.key = $1 AND api_keys.expired_at IS NULL AND accounts.deleted_at IS NULL
        """
        return await connection.fetchval(query, api_key)


async def get_api_key_by_coldkey(coldkey: str, psql_db) -> str | None:
    async with await psql_db.connection() as connection:
        query = """
        SELECT api_keys.key
        FROM api_keys
        JOIN accounts ON api_keys.account_id = accounts.account_id
        WHERE accounts.coldkey = $1
        AND api_keys.expired_at IS NULL
        AND accounts.deleted_at IS NULL
        """
        return await connection.fetchval(query, coldkey)


async def get_validators(
    psql_db: PSQLDB, validator_address: str | None = None, limit: int | None = None
) -> list[tuple[str, str, str]]:
    async with await psql_db.connection() as connection:
        connection: Connection

        if validator_address:
            query = """
            SELECT validator_address, validator_uri, api_key
            FROM validators
            WHERE validator_address = $1
            """
            result = await connection.fetch(query, validator_address)
        else:
            query = """
            SELECT validator_address, validator_uri, api_key
            FROM validators
            """
            if limit:
                query += " LIMIT $1"
                result = await connection.fetch(query, limit)
            else:
                result = await connection.fetch(query)

        return result


async def get_account_validators(account_id: str, psql_db: PSQLDB) -> list[tuple[str]]:
    """
    Fetches the coldkey from the accounts table and all validator_address from the validators table.
    """
    async with await psql_db.connection() as connection:
        connection: Connection
        query = """
        SELECT accounts.coldkey, validators.validator_address
        FROM accounts
        CROSS JOIN validators
        WHERE accounts.account_id = $1
        """
        return await connection.fetch(query, account_id)


async def expire_api_key(api_key: str, psql_db: PSQLDB) -> None:
    async with await psql_db.connection() as connection:
        connection: Connection
        query = """
        UPDATE api_keys
        SET expired_at = CURRENT_TIMESTAMP
        WHERE key = $1
        """
        await connection.execute(query, api_key)


async def get_api_key_data(api_key: str, psql_db: PSQLDB) -> dict | None:
    async with await psql_db.connection() as connection:
        connection: Connection
        query = """
        SELECT key, expired_at
        FROM api_keys
        WHERE key = $1
        """
        return await connection.fetchrow(query, api_key)


async def get_account_info(account_id: str, psql_db: PSQLDB) -> tuple | None:
    async with await psql_db.connection() as connection:
        connection: Connection
        query = """
        SELECT id, account_id, coldkey, created_at, deleted_at
        FROM accounts
        WHERE account_id = $1 AND deleted_at IS NULL
        """
        return await connection.fetchrow(query, account_id)


async def get_existing_api_key(account_id: str, psql_db: PSQLDB) -> str | None:
    async with await psql_db.connection() as connection:
        connection: Connection
        query = """
        SELECT key FROM api_keys
        WHERE account_id = $1 AND expired_at IS NULL
        LIMIT 1
        """
        return await connection.fetchval(query, account_id)


async def get_validators_by_subnet(psql_db: PSQLDB, network: str, subnet: int) -> list[dict]:
    async with await psql_db.connection() as connection:
        connection: Connection
        query = """
        SELECT *
        FROM validators v
        WHERE
            network = $1 and
            subnet = $2
        """

        return await connection.fetch(query, network, subnet)


async def is_account_verified(coldkey: str, psql_db: PSQLDB) -> bool:
    async with await psql_db.connection() as connection:
        connection: Connection
        query = """
        SELECT COUNT(*)
        FROM signing_messages
        WHERE coldkey = $1 AND verified = TRUE
        """
        count = await connection.fetchval(query, coldkey)
        return count > 0


async def get_account_by_coldkey(coldkey: str, psql_db: PSQLDB) -> dict | None:
    async with await psql_db.connection() as connection:
        connection: Connection
        query = """
        SELECT * FROM accounts
        WHERE coldkey = $1 AND deleted_at IS NULL
        LIMIT 1
        """
        return await connection.fetchrow(query, coldkey)


async def expire_signing_message_verified(message: str, coldkey: str, psql_db: PSQLDB) -> None:
    async with await psql_db.connection() as connection:
        connection: Connection
        await connection.execute(
            "UPDATE signing_messages SET expired_at = now(), verified = true WHERE message = $1 and coldkey = $2",
            message,
            coldkey,
        )
