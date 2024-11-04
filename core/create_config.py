import argparse
from typing import Any, Dict

from core.models.config_models import MinerConfig, ValidatorConfig
from core.validators import InputValidators, validate_input


def parse_bool_input(prompt: str, default: bool = False) -> bool:
    result = validate_input(
        f"{prompt} (y/n): (default: {'y' if default else 'n'}) ",
        InputValidators.yes_no,
        default="y" if default else "n"
    )
    return result.lower().startswith("y")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate configuration file")
    parser.add_argument("--dev", action="store_true", help="Use development configuration")
    parser.add_argument("--miner", action="store_true", help="Generate miner configuration")
    return parser.parse_args()


def generate_miner_config(dev: bool = False) -> Dict[str, Any]:
    print("\n🤖 Let's configure your Miner! 🛠️\n")

    network = input("🌐 Enter subtensor network (default: test): ") or "test"
    address = validate_input("🔌 Enter subtensor address (default: None): ", InputValidators.websocket_url) or None

    config = MinerConfig(
        wallet_name=input("\n💼 Enter wallet name (default: default): ") or "default",
        hotkey_name=input("🔑 Enter hotkey name (default: default): ") or "default",
        wandb_token=input("📊 Enter wandb token (default: default): ") or "default",
        huggingface_token=input("🤗 Enter huggingface token (default: default): ") or "default",
        subtensor_network=network,
        subtensor_address=address,
        netuid=176 if network == "test" else 19,
        env="dev" if dev else "prod",
        min_stake_threshold=input(f"Enter MIN_STAKE_THRESHOLD (default: {'0' if network == 'test' else '1000'}): ")
            or ("0" if network == "test" else "1000")
    )

    return vars(config)

def generate_validator_config(dev: bool = False) -> Dict[str, Any]:
    print("\n🎯 Let's set up your Validator! 🚀\n")

    network = input("🌐 Enter subtensor network (default: finney): ") or "finney"
    address = validate_input("🔌 Enter subtensor address (default: None): ", InputValidators.websocket_url) or None

    print("\n📡 GPU Server Configuration")
    gpu_server_input = input("🖥️  Enter GPU server address if you're using one for synth generation: (optional) (default:None)")
    gpu_server = validate_input(gpu_server_input, InputValidators.http_url) if gpu_server_input else None

    print("\n👤 Identity Configuration")
    config = ValidatorConfig(
        wallet_name=input("💼 Enter wallet name (default: default): ") or "default",
        hotkey_name=input("🔑 Enter hotkey name (default: default): ") or "default",
        subtensor_network=network,
        subtensor_address=address,
        netuid=176 if network == "test" else 64,
        env="dev" if dev else "prod",

        print("\n🗄️  Database Configuration")
        postgres_user=input("👤 Enter postgres user (default: user): ") or "user",
        postgres_password=input("🔒 Enter postgres password: "),
        postgres_db=input("📁 Enter postgres database (default: db): ") or "db",
        postgres_host=input("🏠 Enter postgres host (default: localhost): ") or "localhost",
        postgres_port=input("🔌 Enter postgres port (default: 5432): ") or "5432",

        print("\n📦 MinIO Configuration")
        minio_endpoint=input("🎯 Enter minio endpoint: "),
        minio_access_key=input("🔑 Enter minio access key: "),
        minio_secret_key=input("🔐 Enter minio secret key: "),
        gpu_server=gpu_server,
        open_ai_key=input("Enter OpenAI key if you would rather use this for synth") or None,
        api_key=input("Enter Parachutes API if you want to use that for synth generation") or None,
        set_metagraph_weights=parse_bool_input(
            "Set metagraph weights when updated gets really high to not dereg?",
            default=False
        ),
        refresh_nodes=parse_bool_input("Refresh nodes?", default=True) if dev else True,
        localhost=parse_bool_input("Use localhost?", default=True) if dev else False
    )

    return vars(config)

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
    print("\n✨ Welcome to the Configuration Generator! ✨\n")
    print("Let's make your setup process fun and easy! 🎮\n")

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
