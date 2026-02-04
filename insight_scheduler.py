"""
Background scheduler for refreshing pinned charts, insights, and alerts.
"""
import os
import threading
import time
from datetime import datetime
from typing import Callable, Optional, List


def start_insight_scheduler(refresh_callback: Callable[[], None], interval_seconds: int = 900) -> threading.Thread:
    """启动洞察刷新调度器"""
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


def start_alert_scheduler(
    check_callback: Callable[[], None],
    interval_seconds: int = 3600,
    schedule_times: Optional[List[str]] = None,
) -> threading.Thread:
    """
    启动预警/喜报检测调度器
    
    Args:
        check_callback: 检测回调函数
        interval_seconds: 检测间隔（秒），默认1小时
        schedule_times: 固定检测时间点列表，如 ["09:00", "18:00"]
    """
    def _run():
        while True:
            try:
                now = datetime.now()
                
                if schedule_times:
                    # 固定时间点模式
                    current_time = now.strftime("%H:%M")
                    if current_time in schedule_times:
                        print(f"[AlertScheduler] 定时检测触发: {current_time}")
                        check_callback()
                    # 每分钟检查一次
                    time.sleep(60)
                else:
                    # 固定间隔模式
                    check_callback()
                    time.sleep(interval_seconds)
                    
            except Exception as exc:
                print(f"[AlertScheduler] 检测失败: {exc}")
                time.sleep(60)  # 出错后等待1分钟再重试

    worker = threading.Thread(target=_run, daemon=True)
    worker.start()
    print(f"[AlertScheduler] 预警调度器已启动，间隔: {interval_seconds}秒")
    return worker


class AlertSchedulerManager:
    """预警调度器管理器"""
    
    def __init__(self):
        self._thread: Optional[threading.Thread] = None
        self._running = False
        self._check_callback: Optional[Callable] = None
        self._interval = int(os.getenv("ALERT_CHECK_INTERVAL", "3600"))
        self._schedule_times = self._parse_schedule_times()
    
    def _parse_schedule_times(self) -> Optional[List[str]]:
        """解析环境变量中的定时时间点"""
        times_str = os.getenv("ALERT_SCHEDULE_TIMES", "")
        if not times_str:
            return None
        return [t.strip() for t in times_str.split(",") if t.strip()]
    
    def start(self, check_callback: Callable[[], None]):
        """启动调度器"""
        if self._running:
            return
        
        self._check_callback = check_callback
        self._running = True
        self._thread = start_alert_scheduler(
            check_callback=check_callback,
            interval_seconds=self._interval,
            schedule_times=self._schedule_times,
        )
        print(f"[AlertSchedulerManager] 调度器已启动")
    
    def stop(self):
        """停止调度器（注意：由于线程是daemon，实际上会随主进程退出）"""
        self._running = False
        print(f"[AlertSchedulerManager] 调度器已标记停止")
    
    def trigger_now(self):
        """立即触发一次检测"""
        if self._check_callback:
            try:
                self._check_callback()
            except Exception as exc:
                print(f"[AlertSchedulerManager] 手动触发检测失败: {exc}")


# 全局调度器管理器实例
_alert_scheduler_manager: Optional[AlertSchedulerManager] = None


def get_alert_scheduler_manager() -> AlertSchedulerManager:
    """获取预警调度器管理器"""
    global _alert_scheduler_manager
    if _alert_scheduler_manager is None:
        _alert_scheduler_manager = AlertSchedulerManager()
    return _alert_scheduler_manager


def init_alert_scheduler(check_callback: Callable[[], None]) -> AlertSchedulerManager:
    """初始化并启动预警调度器"""
    manager = get_alert_scheduler_manager()
    manager.start(check_callback)
    return manager
