"""
密码加密工具模块
使用 AES-256 对敏感数据进行加密存储
"""
import os
import base64
import hashlib
from typing import Optional

# 尝试导入加密库，如果不存在则使用简单的编码方式
try:
    from cryptography.fernet import Fernet
    from cryptography.hazmat.primitives import hashes
    from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
    CRYPTO_AVAILABLE = True
except ImportError:
    CRYPTO_AVAILABLE = False
    print("[WARNING] cryptography 库未安装，密码将使用 Base64 编码存储（不安全）")
    print("[WARNING] 建议安装: pip install cryptography")


class PasswordEncryption:
    """密码加密/解密工具类"""
    
    def __init__(self, encryption_key: Optional[str] = None):
        """
        初始化加密器
        
        Args:
            encryption_key: 加密密钥，如果不提供则从环境变量 ENCRYPTION_KEY 读取
        """
        self.key = encryption_key or os.getenv('ENCRYPTION_KEY', 'chatbi-default-key-change-in-production')
        self._fernet = None
        
        if CRYPTO_AVAILABLE:
            self._init_fernet()
    
    def _init_fernet(self):
        """初始化 Fernet 加密器"""
        # 使用 PBKDF2 从密钥派生出 Fernet 所需的 32 字节密钥
        salt = b'chatbi_salt_2024'  # 固定 salt，生产环境建议使用随机 salt
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=salt,
            iterations=100000,
        )
        key = base64.urlsafe_b64encode(kdf.derive(self.key.encode()))
        self._fernet = Fernet(key)
    
    def encrypt(self, plaintext: str) -> str:
        """
        加密明文
        
        Args:
            plaintext: 待加密的明文
            
        Returns:
            加密后的密文（Base64 编码）
        """
        if not plaintext:
            return ""
        
        if CRYPTO_AVAILABLE and self._fernet:
            encrypted = self._fernet.encrypt(plaintext.encode())
            return base64.urlsafe_b64encode(encrypted).decode()
        else:
            # 降级方案：Base64 编码（不安全，仅用于开发）
            return base64.urlsafe_b64encode(plaintext.encode()).decode()
    
    def decrypt(self, ciphertext: str) -> str:
        """
        解密密文
        
        Args:
            ciphertext: 加密后的密文
            
        Returns:
            解密后的明文
        """
        if not ciphertext:
            return ""
        
        try:
            if CRYPTO_AVAILABLE and self._fernet:
                encrypted = base64.urlsafe_b64decode(ciphertext.encode())
                decrypted = self._fernet.decrypt(encrypted)
                return decrypted.decode()
            else:
                # 降级方案：Base64 解码
                return base64.urlsafe_b64decode(ciphertext.encode()).decode()
        except Exception as e:
            print(f"[ERROR] 解密失败: {e}")
            # 如果解密失败，可能是明文存储的旧数据，直接返回
            return ciphertext


# 全局加密器实例
_encryptor = None


def get_encryptor() -> PasswordEncryption:
    """获取全局加密器实例"""
    global _encryptor
    if _encryptor is None:
        _encryptor = PasswordEncryption()
    return _encryptor


def encrypt_password(password: str) -> str:
    """加密密码（便捷函数）"""
    return get_encryptor().encrypt(password)


def decrypt_password(encrypted: str) -> str:
    """解密密码（便捷函数）"""
    return get_encryptor().decrypt(encrypted)
