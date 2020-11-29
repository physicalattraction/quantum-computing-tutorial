import json
from typing import Any


def get_secret(key: str) -> Any:
    secrets_file = 'secrets.json'
    with open(secrets_file, 'r') as f:
        secrets = json.load(f)
    return secrets[key]


IBM_TOKEN = get_secret('IBM_TOKEN')
