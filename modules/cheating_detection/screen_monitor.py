import threading
import time
import datetime
import os
import platform
import subprocess
import psutil
import pyautogui
import mss

class ScreenMonitor(threading.Thread):
    def __init__(self, session_data, poll_interval=0.6, screenshot_on_event=True):
        super().__init__()
        self.session_data = session_data
        self.poll_interval = poll_interval
        self.screenshot_on_event = screenshot_on_event
        self.running = False
        self.output_dir = "monitor_logs"
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Allowed window tokens and processes (same as exam_monitor.py)
        self.allowed_window_tokens = ["exam", "proctor", "safe", "secure", "lockdown", "moodle", "canvas"]
        self.allowed_processes = ["chrome.exe", "msedge.exe", "firefox.exe", "brave.exe", "chromium.exe", "safari", "vivaldi.exe"]
        self.browser_processes = ["chrome.exe", "msedge.exe", "firefox.exe", "brave.exe", "chromium.exe", "opera.exe", "safari"]
        
        self.last_title = ""
        self.last_proc = ""
        self.last_screen = self._screen_size()
        
        # For browser change detection
        self.last_browsers_set = set()
        self.last_multiple_browser_log_time = 0
        self.multiple_browser_log_interval = 60  # seconds
    
    def _screen_size(self):
        try:
            if pyautogui:
                s = pyautogui.size()
                return (s.width, s.height)
            elif mss:
                with mss.mss() as s:
                    mon = s.monitors[0]
                    return (mon["width"], mon["height"])
        except:
            return None
    
    def _get_active_window_info(self):
        system = platform.system()
        title = ""
        pname = ""
        if system == "Windows":
            try:
                import win32gui, win32process
                hwnd = win32gui.GetForegroundWindow()
                title = win32gui.GetWindowText(hwnd) or ""
                _, pid = win32process.GetWindowThreadProcessId(hwnd)
                pname = psutil.Process(pid).name().lower()
            except:
                pass
        elif system == "Darwin":
            try:
                script = 'tell application "System Events"\nset frontApp to name of first application process whose frontmost is true\nend tell\n' \
                         'tell application frontApp\ntry\nset w to name of front window\non error\nset w to ""\nend try\nend tell\n' \
                         'return frontApp & "||" & w'
                p = subprocess.run(["osascript", "-e", script], capture_output=True, text=True)
                out = p.stdout.strip()
                if "||" in out:
                    app, title = out.split("||", 1)
                else:
                    app = out
                pname = app.lower()
            except:
                pass
        else:  # Linux
            try:
                p = subprocess.run(["xdotool", "getwindowfocus", "getwindowname"], capture_output=True, text=True, timeout=0.5)
                title = p.stdout.strip()
            except:
                pass
            try:
                p = subprocess.run(["xdotool", "getwindowfocus", "getwindowpid"], capture_output=True, text=True, timeout=0.5)
                pid = p.stdout.strip()
                if pid.isdigit():
                    pname = psutil.Process(int(pid)).name().lower()
            except:
                pass
        return title, pname
    
    def _take_screenshot(self, prefix="event"):
        fname = os.path.join(self.output_dir, f"{prefix}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.png")
        try:
            if mss:
                with mss.mss() as s:
                    sshot = s.grab(s.monitors[0])
                    mss.tools.to_png(sshot.rgb, sshot.size, output=fname)
            elif pyautogui:
                img = pyautogui.screenshot()
                img.save(fname)
            else:
                return None
            return fname
        except:
            return None
    
    def run(self):
        self.running = True
        while self.running:
            try:
                title, proc = self._get_active_window_info()
                ts = time.time()
                
                # Window change detection
                if title != self.last_title or proc != self.last_proc:
                    event = {
                        "timestamp": ts,
                        "type": "window_change",
                        "title": title,
                        "process": proc,
                        "old_title": self.last_title,
                        "old_proc": self.last_proc
                    }
                    # Check allowed
                    allowed = False
                    low_title = (title or "").lower()
                    if any(tok in low_title for tok in self.allowed_window_tokens):
                        allowed = True
                    if proc and proc.lower() in [p.lower() for p in self.allowed_processes]:
                        allowed = True
                    if not allowed:
                        event["suspicious"] = True
                        if self.screenshot_on_event:
                            event["screenshot"] = self._take_screenshot("window_switch")
                    else:
                        event["suspicious"] = False
                    self.session_data.add_screen_event(event)
                    self.last_title, self.last_proc = title, proc
                
                # Browser process monitoring – log only when set changes, and at most every 60 seconds
                try:
                    procs = [p.info for p in psutil.process_iter(['pid', 'name'])]
                    now = time.time()
                    current_browsers = set()
                    for p in procs:
                        name = (p.get('name') or "").lower()
                        if name in self.browser_processes:
                            current_browsers.add(name)
                    
                    # If the set of browsers changed
                    if current_browsers != self.last_browsers_set:
                        self.last_browsers_set = current_browsers
                        if len(current_browsers) > 1:
                            event = {
                                "timestamp": ts,
                                "type": "multiple_browsers",
                                "browsers": list(current_browsers),
                                "suspicious": True
                            }
                            if self.screenshot_on_event:
                                event["screenshot"] = self._take_screenshot("multiple_browsers")
                            self.session_data.add_screen_event(event)
                            self.last_multiple_browser_log_time = now
                    # If still multiple browsers but no change, log only every 60 seconds
                    elif len(current_browsers) > 1 and (now - self.last_multiple_browser_log_time) >= self.multiple_browser_log_interval:
                        event = {
                            "timestamp": ts,
                            "type": "multiple_browsers",
                            "browsers": list(current_browsers),
                            "suspicious": True
                        }
                        if self.screenshot_on_event:
                            event["screenshot"] = self._take_screenshot("multiple_browsers")
                        self.session_data.add_screen_event(event)
                        self.last_multiple_browser_log_time = now
                except Exception as e:
                    print(f"Browser monitoring error: {e}")
                
                # Screen size change
                cur_screen = self._screen_size()
                if cur_screen != self.last_screen and cur_screen is not None:
                    event = {
                        "timestamp": ts,
                        "type": "screen_size_change",
                        "new_size": cur_screen,
                        "old_size": self.last_screen,
                        "suspicious": True
                    }
                    if self.screenshot_on_event:
                        event["screenshot"] = self._take_screenshot("screen_change")
                    self.session_data.add_screen_event(event)
                    self.last_screen = cur_screen
                
                time.sleep(self.poll_interval)
            except Exception as e:
                print(f"Screen monitor error: {e}")
                time.sleep(self.poll_interval)
    
    def stop(self):
        self.running = False