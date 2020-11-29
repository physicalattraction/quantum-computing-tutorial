import json
from typing import Any
import os.path

def get_secret(key: str) -> Any:
    settings_dir = os.path.dirname(__file__)
    secrets_file = os.path.join(settings_dir, 'secrets.json')
    with open(secrets_file, 'r') as f:
        secrets = json.load(f)
    return secrets[key]


IBM_TOKEN = get_secret('IBM_TOKEN')
