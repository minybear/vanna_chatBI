"""预警/喜报疲劳度：同一图表同类型在冷却期内不重复发送。"""
import os
import json
import threading
from datetime import datetime, timedelta
from typing import Optional


_LOCK = threading.Lock()
_DATA: Optional[dict] = None


def _log_path() -> str:
    path = os.getenv("ALERT_FATIGUE_LOG_PATH", "data/alert_sent_log.json")
    if not os.path.isabs(path):
        base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        path = os.path.join(base, path)
    return path


def _load() -> dict:
    global _DATA
    with _LOCK:
        if _DATA is not None:
            return _DATA
        p = _log_path()
        if os.path.isfile(p):
            try:
                with open(p, "r", encoding="utf-8") as f:
                    _DATA = json.load(f)
            except Exception:
                _DATA = {"last_sent": {}, "updated_at": None}
        else:
            _DATA = {"last_sent": {}, "updated_at": None}
        return _DATA


def _save() -> None:
    global _DATA
    with _LOCK:
        if _DATA is None:
            return
        p = _log_path()
        d = os.path.dirname(p)
        if d and not os.path.isdir(d):
            os.makedirs(d, exist_ok=True)
        _DATA["updated_at"] = datetime.now().isoformat()
        try:
            with open(p, "w", encoding="utf-8") as f:
                json.dump(_DATA, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"[AlertFatigue] 写入失败: {e}")


def _key(operator_id: str, chart_id: str, notification_type: str) -> str:
    return f"{operator_id or 'default'}|{chart_id or ''}|{notification_type}"


def should_send(
    operator_id: str,
    chart_id: str,
    notification_type: str,
    effective_cooldown_hours: float,
) -> bool:
    """
    是否允许发送（未在疲劳期内）。
    effective_cooldown_hours <= 0 表示不限制，始终允许发送。
    """
    if effective_cooldown_hours <= 0:
        return True
    if not chart_id:
        return True
    data = _load()
    last_sent = (data.get("last_sent") or {}).get(
        _key(operator_id, chart_id, notification_type)
    )
    if not last_sent:
        return True
    try:
        t = datetime.fromisoformat(last_sent.replace("Z", "+00:00"))
        if t.tzinfo:
            t = t.astimezone().replace(tzinfo=None)
    except Exception:
        return True
    return datetime.now() - t >= timedelta(hours=effective_cooldown_hours)


def record_sent(
    operator_id: str,
    chart_id: str,
    notification_type: str,
) -> None:
    """记录本次发送时间。"""
    if not chart_id:
        return
    data = _load()
    if "last_sent" not in data:
        data["last_sent"] = {}
    data["last_sent"][_key(operator_id, chart_id, notification_type)] = (
        datetime.now().isoformat()
    )
    _save()


def get_effective_cooldown_hours(
    user_settings: dict,
    label: Optional[dict],
) -> float:
    """
    计算有效冷却小时数。
    - 单图表 label.fatigue_cooldown_hours 不为 None 时：0=不限制，>0=该值（小时）
    - 否则用全局：user_settings.alert_fatigue_enabled 为 False 或未设置则 0，否则 alert_fatigue_cooldown_hours
    """
    if label is not None:
        v = label.get("fatigue_cooldown_hours")
        if v is not None:
            try:
                h = float(v)
                return 0.0 if h <= 0 else h
            except (TypeError, ValueError):
                pass
    if not user_settings.get("alert_fatigue_enabled", True):
        return 0.0
    try:
        h = float(user_settings.get("alert_fatigue_cooldown_hours", 24))
        return 0.0 if h <= 0 else h
    except (TypeError, ValueError):
        return 24.0
