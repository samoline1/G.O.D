import argparse
import random
import re
import secrets
import string
from typing import Any
from typing import Callable


# Lots of this was taken from sn19, some of which might not be needed, will come back to clean one we have an end2end
# right now we're not using any of this - just here for shitz and gigz apparently


def generate_secure_password(length: int = 16) -> str:
    alphabet = string.ascii_letters + string.digits
    password = [
        secrets.choice(string.ascii_uppercase),
        secrets.choice(string.ascii_lowercase),
        secrets.choice(string.digits),
    ]
    password += [secrets.choice(alphabet) for _ in range(length - 3)]
    password = list(password)  # Convert to list for shuffling
    random.shuffle(password)  # Use random.shuffle instead of secrets.shuffle
    return "".join(password)


def validate_input(prompt: str, validator: Callable[[str], bool], default: str | None = None) -> str:
    while True:
        value = input(prompt)
        if validator(value):
            return value
        elif default and not value:
            return default
        print("Invalid input. Please try again.")


def yes_no_validator(value: str) -> bool:
    return value.lower() in ["y", "n", "yes", "no"] or not value


def non_empty_bool(value: str) -> bool:
    return bool(value.strip())


def number_validator(value: str) -> bool:
    return value.isdigit()


def float_validator(value: str) -> bool:
    try:
        float(value)
        return True
    except ValueError:
        return False


def websocket_validator(value: str | None) -> bool:
    if not value:
        return True
    return re.match(r"^wss?://", value) is not None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate configuration file")
    parser.add_argument("--dev", action="store_true", help="Use development configuration")
    parser.add_argument("--miner", action="store_true", help="Generate miner configuration")
    return parser.parse_args()


def generate_miner_config(dev: bool = False) -> dict[str, Any]:
    config: dict[str, Any] = {}
    config["WALLET_NAME"] = input("Enter wallet name (default: default): ") or "default"
    config["HOTKEY_NAME"] = input("Enter hotkey name (default: default): ") or "default"
    config["WANDB_TOKEN"] = input("Enter wandb token (default: default): ") or "default"
    config["HUGGINGFACE_TOKEN"] = input("Enter huggingface token (default: default): ") or "default"
    config["SUBTENSOR_NETWORK"] = input("Enter subtensor network (default: test): ") or "test"
    address = validate_input("Enter subtensor address (default: None): ", websocket_validator) or None
    if address:
        config["SUBTENSOR_ADDRESS"]  = address
    default_stake_threshold = "0" if config["SUBTENSOR_NETWORK"] == "test" else "1000"
    config["NETUID"] = 176 if config["SUBTENSOR_NETWORK"] == "test" else 19
    config["ENV"] = "dev" if dev else "prod"
    config["IS_VALIDATOR"] = "False"
#    config["NODE_PORT"] = input("Enter NODE_PORT (default: 4002): ") or "4002"
#    config["NODE_EXTERNAL_IP"] = input("Enter NODE_EXTERNAL_IP (leave blank if not needed): ")
    config["MIN_STAKE_THRESHOLD"] = input("Enter MIN_STAKE_THRESHOLD (default: 1000): ") or default_stake_threshold
    config["REFRESH_NODES"] = "true"
    return config

def generate_validator_config(dev: bool = False) -> dict[str, Any]:
    config: dict[str, Any] = {}
    config["WALLET_NAME"] = input("Enter wallet name (default: default): ") or "default"
    config["HOTKEY_NAME"] = input("Enter hotkey name (default: default): ") or "default"
    config["SUBTENSOR_NETWORK"] = input("Enter subtensor network (default: finney): ") or "finney"
    address = validate_input("Enter subtensor address (default: None): ", websocket_validator) or None
    config["POSTGRES_USER"] = input("Enter postgres user (default: user): ") or "user"
    config["POSTGRES_PASSWORD"] = input("Enter postgres password ")
    config["POSTGRES_DB"] = input("Enter postgres database (default: db): ") or "db"
    config["POSTGRES_HOST"] = input("Enter postgres host (default: localhost): ") or "localhost"
    config["POSTGRES_PORT"] = input("Enter postgres port (default: 5432): ") or "5432"
    config["MINIO_ENDPOINT"] = input("Enter minio endpoint")
    config["MINIO_ACCESS_KEY"] = input("Enter minio access key")
    config["MINIO_SECRET_KEY"] = input("Enter minio secret key")
    config["CORCEL_TOKEN"] = input("Enter corcel token if you're planning to use this  for synth gen") or None
    config["OPEN_AI"] = input("Enter OpenAI key if you would rather use this for synth") or None
    config["API_KEY"] = input("Enter Parachutes API if you want to use that for synth generation") or None
    if address:
        config["SUBTENSOR_ADDRESS"]  = address
    config["NETUID"] = 176 if config["SUBTENSOR_NETWORK"] == "test" else 64
    gpu_server = input(
                "Enter GPU server address if you're using one for synth generation: (optional) (default:None)"
        )
    if gpu_server:
            config['GPU_SERVER'] = validate_input(
                gpu_server,
                lambda x: x == "" or re.match(r"^https?://.+", x) is not None
            )
    config["SET_METAGRAPH_WEIGHTS_WITH_HIGH_UPDATED_TO_NOT_DEREG"] = (
        "true"
        if validate_input(
            "Set metagraph weights when updated gets really high to not dereg? (y/n): (default: n)",
            yes_no_validator,
            default="n",
        )
        .lower()
        .startswith("y")
        else "false"
    )

    if dev:
        config["ENV"] = "dev"
        config["REFRESH_NODES"] = (
            "true" if validate_input("Refresh nodes? (y/n): (default: y)", yes_no_validator).lower().startswith("y") else "false"
        )
        config["LOCALHOST"] = (
            "true" if validate_input("Use localhost? (y/n): (default: y)", yes_no_validator).lower().startswith("y") else "false"
        )
    else:
        config["ENV"] = "prod"

    config["ENV_FILE"] = ".vali.env"
    return config

def generate_config(dev: bool = False, miner: bool = False) -> dict[str, Any]:
    if miner:
        return generate_miner_config(dev)
    else:
        return generate_validator_config(dev)


def write_config_to_file(config: dict[str, Any], env: str) -> None:
    filename = f".{env}.env"
    with open(filename, "w") as f:
        for key, value in config.items():
            f.write(f"{key}={value}\n")


if __name__ == "__main__":
    args = parse_args()
    print("Welcome to the configuration generator!")

    if args.miner:
        config = generate_config(miner=True)
        name = "1"

    else:
        env = "dev" if args.dev else "prod"
        config = generate_config(dev=args.dev)
        name = "vali"

    write_config_to_file(config, name)
    print(f"Configuration has been written to .{name}.env")
    if not args.miner:
        print("Please make sure to keep your database credentials secure.")
