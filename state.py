import threading
from typing import Any, Dict, List


config_lock = threading.Lock()
parking_lock = threading.Lock()
events_lock = threading.Lock()

exit_camera_source = None
parking_inside: Dict[str, float] = {}
recent_events: List[Dict[str, Any]] = []


def set_exit_camera_source(value: str) -> None:
    global exit_camera_source
    with config_lock:
        exit_camera_source = value


def get_exit_camera_source() -> str:
    with config_lock:
        return exit_camera_source


def push_event(event: Dict[str, Any]) -> None:
    with events_lock:
        recent_events.insert(0, event)
        del recent_events[200:]
