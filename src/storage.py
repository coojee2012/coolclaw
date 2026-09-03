import os
import json
import base64
import hashlib
from typing import Optional, Any
from pathlib import Path
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC


class SecureStorage:
    _instance = None
    _cipher: Optional[Fernet] = None
    _storage_path: Path = Path.home() / ".opencode_helper" / "secrets.enc"
    _key_file: Path = Path.home() / ".opencode_helper" / ".key"

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialize()
        return cls._instance

    def _initialize(self):
        self._storage_path.parent.mkdir(parents=True, exist_ok=True)
        self._load_or_create_key()

    def _get_key_from_password(self, password: str) -> bytes:
        salt = b"opencode_helper_salt_v1"
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=salt,
            iterations=100000,
        )
        key = base64.urlsafe_b64encode(kdf.derive(password.encode()))
        return key

    def _load_or_create_key(self):
        if self._key_file.exists():
            try:
                with open(self._key_file, "rb") as f:
                    encrypted_key = f.read()
                master_password = os.environ.get(
                    "OPENCODE_MASTER_PASSWORD", "default_password_change_me"
                )
                key = self._get_key_from_password(master_password)
                self._cipher = Fernet(key)
                try:
                    decrypted = self._cipher.decrypt(encrypted_key)
                    self._storage_key = decrypted
                except:
                    self._storage_key = Fernet.generate_key()
                    encrypted = self._cipher.encrypt(self._storage_key)
                    with open(self._key_file, "wb") as f:
                        f.write(encrypted)
            except Exception:
                self._storage_key = Fernet.generate_key()
                self._cipher = Fernet(self._storage_key)
        else:
            self._storage_key = Fernet.generate_key()
            self._cipher = Fernet(self._storage_key)
            master_password = os.environ.get(
                "OPENCODE_MASTER_PASSWORD", "default_password_change_me"
            )
            key = self._get_key_from_password(master_password)
            cipher_temp = Fernet(key)
            encrypted = cipher_temp.encrypt(self._storage_key)
            with open(self._key_file, "wb") as f:
                f.write(encrypted)
            os.chmod(str(self._key_file), 0o600)

    def set(self, key: str, value: Any) -> bool:
        try:
            data = self._load_all()
            serialized = json.dumps(value)
            encrypted = self._cipher.encrypt(serialized.encode())
            data[key] = base64.b64encode(encrypted).decode()
            self._save_all(data)
            return True
        except Exception as e:
            print(f"SecureStorage set error: {e}")
            return False

    def get(self, key: str, default: Any = None) -> Any:
        try:
            data = self._load_all()
            if key in data:
                encrypted = base64.b64decode(data[key])
                decrypted = self._cipher.decrypt(encrypted)
                return json.loads(decrypted.decode())
            return default
        except Exception as e:
            print(f"SecureStorage get error: {e}")
            return default

    def delete(self, key: str) -> bool:
        try:
            data = self._load_all()
            if key in data:
                del data[key]
                self._save_all(data)
            return True
        except Exception:
            return False

    def _load_all(self) -> dict:
        if self._storage_path.exists():
            try:
                with open(self._storage_path, "r") as f:
                    return json.load(f)
            except:
                return {}
        return {}

    def _save_all(self, data: dict):
        with open(self._storage_path, "w") as f:
            json.dump(data, f)
        os.chmod(str(self._storage_path), 0o600)

    def list_keys(self) -> list[str]:
        return list(self._load_all().keys())

    def has(self, key: str) -> bool:
        return key in self._load_all()


secrets = SecureStorage()
