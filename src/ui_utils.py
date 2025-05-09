"""
UI utilities for the CFA MCQ Reproducer project.

Provides loading animations and other UI elements for the command line interface.
"""
import sys
import time
import threading
import itertools
from typing import Optional, Callable


class LoadingAnimation:
    """
    A class to display a loading animation in the console.
    
    Attributes:
        message (str): The message to display alongside the animation.
        animation_chars (list): Characters to use for the animation.
        delay (float): Delay between animation frames in seconds.
        _stop_event (threading.Event): Event to signal the animation thread to stop.
        _thread (threading.Thread): Thread running the animation.
    """
    
    def __init__(self, message: str = "Processing", animation_chars: Optional[list] = None, 
                 delay: float = 0.1):
        """
        Initialize the loading animation.
        
        Args:
            message: Message to display alongside the animation.
            animation_chars: Characters to use for the animation. Defaults to a spinner.
            delay: Delay between animation frames in seconds.
        """
        self.message = message
        self.animation_chars = animation_chars or ['⠋', '⠙', '⠹', '⠸', '⠼', '⠴', '⠦', '⠧', '⠇', '⠏']
        self.delay = delay
        self._stop_event = threading.Event()
        self._thread = None
        self._progress = None
        self._total = None
        
    def _animate(self):
        """Animation loop that runs in a separate thread."""
        for char in itertools.cycle(self.animation_chars):
            if self._stop_event.is_set():
                break
                
            if self._progress is not None and self._total is not None:
                progress_str = f" [{self._progress}/{self._total}]"
            else:
                progress_str = ""
                
            sys.stdout.write(f"\r{self.message}{progress_str} {char} ")
            sys.stdout.flush()
            time.sleep(self.delay)
            
    def start(self):
        """Start the loading animation in a separate thread."""
        if self._thread is not None and self._thread.is_alive():
            return
            
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._animate)
        self._thread.daemon = True
        self._thread.start()
        
    def stop(self, clear_line: bool = True):
        """
        Stop the loading animation.
        
        Args:
            clear_line: Whether to clear the line after stopping.
        """
        if self._thread is None or not self._thread.is_alive():
            return 
            
        self._stop_event.set()
        self._thread.join()
        
        if clear_line:
            sys.stdout.write("\r" + " " * (len(self.message) + 20) + "\r")
            sys.stdout.flush()
            
    def update_progress(self, current: int, total: int):
        """
        Update the progress display.
        
        Args:
            current: Current progress value.
            total: Total items to process.
        """
        self._progress = current
        self._total = total


def run_with_loading_animation(func: Callable, *args, message: str = "Processing", **kwargs):
    """
    Run a function with a loading animation.
    
    Args:
        func: Function to run.
        *args: Arguments to pass to the function.
        message: Message to display alongside the animation.
        **kwargs: Keyword arguments to pass to the function.
        
    Returns:
        The result of the function.
    """
    animation = LoadingAnimation(message=message)
    animation.start()
    
    try:
        result = func(*args, **kwargs)
        return result
    finally:
        animation.stop()


def print_success(message: str):
    """
    Print a success message with a checkmark.
    
    Args:
        message: Message to print.
    """
    print(f"\033[92m✓ {message}\033[0m")


def print_error(message: str):
    """
    Print an error message with an X.
    
    Args:
        message: Message to print.
    """
    print(f"\033[91m✗ {message}\033[0m")


def print_info(message: str):
    """
    Print an info message with an info symbol.
    
    Args:
        message: Message to print.
    """
    print(f"\033[94mℹ {message}\033[0m")


def print_warning(message: str):
    """
    Print a warning message with a warning symbol.
    
    Args:
        message: Message to print.
    """
    print(f"\033[93m⚠ {message}\033[0m")
