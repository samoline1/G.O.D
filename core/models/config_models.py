from dataclasses import dataclass
from typing import Optional

@dataclass
class BaseConfig:
    wallet_name: str
    hotkey_name: str
    subtensor_network: str
    subtensor_address: Optional[str]
    netuid: int
    env: str
    refresh_nodes: bool = True

class MinerConfig(BaseConfig):
    wandb_token: str
    huggingface_token: str
    min_stake_threshold: str
    is_validator: bool = False

class ValidatorConfig(BaseConfig):
    postgres_user: str
    postgres_password: str
    postgres_db: str
    postgres_host: str
    postgres_port: str
    minio_endpoint: str
    minio_access_key: str
    minio_secret_key: str
    set_metagraph_weights: bool
    gpu_server: Optional[str] = None
    open_ai_key: Optional[str] = None
    api_key: Optional[str] = None
    localhost: bool = False
    env_file: str = ".vali.env"
