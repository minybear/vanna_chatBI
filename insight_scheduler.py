"""
Background scheduler for refreshing pinned charts and insights.
"""
import threading
import time
from typing import Callable


def start_insight_scheduler(refresh_callback: Callable[[], None], interval_seconds: int = 900) -> threading.Thread:
    def _run():
        while True:
            try:
                refresh_callback()
            except Exception as exc:
                print(f"[AnalyticsScheduler] 刷新失败: {exc}")
            time.sleep(interval_seconds)

    worker = threading.Thread(target=_run, daemon=True)
    worker.start()
    return worker
