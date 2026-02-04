"""认证、JWT、租户隔离。"""
import re
import json
import secrets
import jwt
from datetime import datetime, timedelta
from typing import Optional, Dict, Any

from fastapi import Response, HTTPException, Header, Depends

from core.config import (
    AUTH_ENABLED,
    AUTH_USERS_JSON,
    AUTH_TOKEN_TTL_SECONDS,
    AUTH_TOKEN_SLIDING_ENABLED,
    AUTH_TOKEN_REFRESH_SECONDS,
    JWT_SECRET,
    JWT_ALGORITHM,
    TENANT_ISOLATION_ENABLED,
    TENANT_COLUMN,
)

_AUTH_USERS_CACHE: Optional[Dict[str, Dict[str, Any]]] = None


def _load_auth_users() -> Dict[str, Dict[str, Any]]:
    if not AUTH_USERS_JSON:
        return {}
    try:
        raw = json.loads(AUTH_USERS_JSON)
    except json.JSONDecodeError as exc:
        raise ValueError(f"AUTH_USERS_JSON 解析失败: {exc}") from exc

    users: Dict[str, Dict[str, Any]] = {}
    if isinstance(raw, list):
        items = raw
    elif isinstance(raw, dict):
        items = list(raw.values())
    else:
        raise ValueError("AUTH_USERS_JSON 必须是数组或对象")

    for item in items:
        if not isinstance(item, dict):
            continue
        username = str(item.get("username", "")).strip()
        password = str(item.get("password", "")).strip()
        operator_id = str(item.get("operator_id", "")).strip()
        if not username or not password or not operator_id:
            raise ValueError("AUTH_USERS_JSON 中每个用户必须包含 username/password/operator_id")
        users[username] = {
            "username": username,
            "password": password,
            "operator_id": operator_id,
        }
    return users


def _get_auth_users() -> Dict[str, Dict[str, Any]]:
    global _AUTH_USERS_CACHE
    if _AUTH_USERS_CACHE is None:
        _AUTH_USERS_CACHE = _load_auth_users()
    return _AUTH_USERS_CACHE


def _authenticate_user(username: str, password: str) -> Optional[Dict[str, Any]]:
    users = _get_auth_users()
    user = users.get(username)
    if not user:
        return None
    if secrets.compare_digest(user.get("password", ""), password):
        return {
            "username": user["username"],
            "operator_id": user["operator_id"],
        }
    return None


def _issue_token(user: Dict[str, Any]) -> str:
    payload = {
        "username": user["username"],
        "operator_id": user["operator_id"],
        "exp": datetime.utcnow() + timedelta(seconds=AUTH_TOKEN_TTL_SECONDS),
        "iat": datetime.utcnow(),
    }
    return jwt.encode(payload, JWT_SECRET, algorithm=JWT_ALGORITHM)


def _parse_bearer_token(authorization: Optional[str]) -> Optional[str]:
    if not authorization:
        return None
    parts = authorization.strip().split(" ", 1)
    if len(parts) == 2 and parts[0].lower() == "bearer":
        return parts[1].strip()
    return None


def _maybe_refresh_token(payload: Dict[str, Any], response: Response) -> None:
    if not AUTH_TOKEN_SLIDING_ENABLED:
        return
    exp_ts = payload.get("exp")
    if not exp_ts:
        return
    try:
        exp_ts = int(exp_ts)
    except Exception:
        return
    now_ts = int(datetime.utcnow().timestamp())
    remaining = exp_ts - now_ts
    if remaining > AUTH_TOKEN_REFRESH_SECONDS:
        return
    refreshed = _issue_token({
        "username": payload.get("username"),
        "operator_id": payload.get("operator_id"),
    })
    response.headers["X-Auth-Token"] = refreshed
    response.headers["X-Auth-Expires-In"] = str(AUTH_TOKEN_TTL_SECONDS)


def get_current_user(
    response: Response,
    authorization: Optional[str] = Header(None),
    x_auth_token: Optional[str] = Header(None),
) -> Dict[str, Any]:
    if not AUTH_ENABLED:
        return {"username": "anonymous", "operator_id": ""}

    token = x_auth_token or _parse_bearer_token(authorization)
    if not token:
        raise HTTPException(status_code=401, detail="未登录或缺少访问令牌")

    try:
        payload = jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALGORITHM])
        _maybe_refresh_token(payload, response)
        return {
            "username": payload.get("username"),
            "operator_id": payload.get("operator_id"),
        }
    except jwt.ExpiredSignatureError:
        raise HTTPException(status_code=401, detail="访问令牌已过期")
    except jwt.InvalidTokenError:
        raise HTTPException(status_code=401, detail="访问令牌无效")


def _has_tenant_filter(sql: str, tenant_column: str, operator_id: str) -> bool:
    if not sql:
        return False
    pattern = rf"\b{re.escape(tenant_column)}\b\s*=\s*'{re.escape(operator_id)}'"
    return re.search(pattern, sql, re.IGNORECASE) is not None


def _inject_tenant_filter(sql: str, tenant_column: str, operator_id: str) -> str:
    sql_stripped = sql.strip()
    if not sql_stripped:
        return sql_stripped

    has_semicolon = sql_stripped.endswith(";")
    if has_semicolon:
        sql_stripped = sql_stripped[:-1].rstrip()

    sql_upper = sql_stripped.upper()
    if "WHERE" in sql_upper:
        replaced = re.sub(
            r"\bWHERE\b",
            f"WHERE {tenant_column} = '{operator_id}' AND ",
            sql_stripped,
            count=1,
            flags=re.IGNORECASE,
        )
        return f"{replaced};" if has_semicolon else replaced

    insert_pos = len(sql_stripped)
    for keyword in [" GROUP BY", " ORDER BY", " LIMIT", " HAVING", " UNION ", " INTERSECT ", " EXCEPT "]:
        pos = sql_upper.find(keyword)
        if pos != -1 and pos < insert_pos:
            insert_pos = pos

    if insert_pos == len(sql_stripped):
        injected = f"{sql_stripped} WHERE {tenant_column} = '{operator_id}'"
    else:
        before = sql_stripped[:insert_pos].rstrip()
        after = sql_stripped[insert_pos:]
        injected = f"{before} WHERE {tenant_column} = '{operator_id}' {after.lstrip()}"

    return f"{injected};" if has_semicolon else injected


def transform_sql_with_tenant(sql: str, operator_id: str) -> str:
    if not TENANT_ISOLATION_ENABLED:
        return sql
    if not operator_id:
        raise ValueError("租户隔离已启用，但 operator_id 为空")
    if _has_tenant_filter(sql, TENANT_COLUMN, operator_id):
        return sql
    return _inject_tenant_filter(sql, TENANT_COLUMN, operator_id)


def require_operator_id(current_user: Dict[str, Any]) -> str:
    operator_id = (current_user or {}).get("operator_id", "")
    if TENANT_ISOLATION_ENABLED and not operator_id:
        raise HTTPException(status_code=401, detail="租户隔离已启用，但未绑定 operator_id")
    return operator_id
