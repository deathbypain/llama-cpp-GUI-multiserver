"""System tray icon management (shared across UIs)."""

from typing import Optional, TYPE_CHECKING, Any

if TYPE_CHECKING:
    import pystray

try:
    import pystray
    from pystray import MenuItem as item
    PYSTRAY_AVAILABLE = True
except ImportError:
    pystray = None  
    PYSTRAY_AVAILABLE = False

from PIL import Image
import threading

from .utils import resource_path

TRAY_AVAILABLE = PYSTRAY_AVAILABLE

class TrayManager:
    """Manages system tray icon and menu callbacks.

    The controller object must implement:
      - show_window()
      - open_browser()
      - quit_application()
      - any_server_running()
    """

    def __init__(self, controller):
        self.controller = controller
        self.tray_icon: Optional[Any] = None  # Use Any as fallback

    def create_tray_icon(self):
        if not TRAY_AVAILABLE:
            return None

        image = self._load_icon()
        menu_items = [
            item("Show Window", self._on_show, default=True),
            item("Open Browser", self._on_browser, enabled=lambda i: self.controller.any_server_running()),
            pystray.Menu.SEPARATOR,
            item("Quit Application", self._on_quit),
        ]
        return pystray.Icon("llama_server", image, "LLaMA Server", menu=pystray.Menu(*menu_items))

    def show_tray(self):
        if not TRAY_AVAILABLE:
            return
        if self.tray_icon is None:
            self.tray_icon = self.create_tray_icon()
            threading.Thread(target=self.tray_icon.run, daemon=True).start()

    def hide_tray(self):
        if self.tray_icon:
            self.tray_icon.stop()
            self.tray_icon = None

    def _load_icon(self):
        try:
            import os
            core_dir = os.path.dirname(os.path.abspath(__file__))
            root_dir = os.path.dirname(core_dir)
            icon_path = os.path.join(root_dir, "assets", "llama-cpp.ico")
            return Image.open(icon_path)
        except Exception:
            return Image.new("RGB", (64, 64), color=(0, 0, 0))

    def _on_show(self, icon=None, item=None):
        self.controller.show_window()

    def _on_browser(self, icon=None, item=None):
        self.controller.open_browser()

    def _on_quit(self, icon=None, item=None):
        self.controller.quit_application()