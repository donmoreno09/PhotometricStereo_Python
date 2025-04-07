"""
Progress reporting tools for long-running operations.
Python implementation of ShowProgress.m
"""

import time
import sys
from PyQt5.QtCore import QObject, pyqtSignal, QEventLoop, QTimer
from PyQt5.QtWidgets import QProgressBar, QApplication


class ProgressReporter:
    """Base class for progress reporting."""
    
    def __init__(self, total_steps=100, title="Processing"):
        """
        Initialize progress reporter.
        
        Parameters
        ----------
        total_steps : int
            Total number of steps in the operation
        title : str
            Title of the operation
        """
        self.total_steps = total_steps
        self.title = title
        self.current_step = 0
        self.start_time = time.time()
    
    def update(self, step=None, message=None):
        """
        Update progress.
        
        Parameters
        ----------
        step : int, optional
            Current step. If None, increment by 1
        message : str, optional
            Message to display
        """
        if step is not None:
            self.current_step = step
        else:
            self.current_step += 1
        
        # Ensure step is within bounds
        self.current_step = max(0, min(self.current_step, self.total_steps))
        
        # Calculate percentage
        percentage = (self.current_step / self.total_steps) * 100 if self.total_steps > 0 else 0
        
        self._update_impl(percentage, message)
    
    def _update_impl(self, percentage, message):
        """Implementation-specific update method."""
        pass
    
    def finish(self, message="Operation completed"):
        """
        Mark operation as completed.
        
        Parameters
        ----------
        message : str, optional
            Completion message
        """
        elapsed_time = time.time() - self.start_time
        self._finish_impl(elapsed_time, message)
    
    def _finish_impl(self, elapsed_time, message):
        """Implementation-specific finish method."""
        pass


class ConsoleProgressReporter(ProgressReporter):
    """Progress reporter that outputs to console."""
    
    def __init__(self, total_steps=100, title="Processing", width=50):
        """
        Initialize console progress reporter.
        
        Parameters
        ----------
        total_steps : int
            Total number of steps in the operation
        title : str
            Title of the operation
        width : int
            Width of progress bar in characters
        """
        super().__init__(total_steps, title)
        self.width = width
        self.last_percentage = -1
        
        # Print initial bar
        sys.stdout.write(f"{self.title}: [" + " " * self.width + "] 0%\r")
        sys.stdout.flush()
    
    def _update_impl(self, percentage, message):
        """Update console progress bar."""
        # Only update if percentage has changed significantly (avoid excessive printing)
        if int(percentage) > self.last_percentage:
            self.last_percentage = int(percentage)
            
            # Calculate bar width
            filled_width = int(self.width * percentage / 100)
            bar = "=" * filled_width + " " * (self.width - filled_width)
            
            # Print progress bar
            if message:
                sys.stdout.write(f"{self.title}: [{bar}] {percentage:.1f}% - {message}\r")
            else:
                sys.stdout.write(f"{self.title}: [{bar}] {percentage:.1f}%\r")
            
            sys.stdout.flush()
    
    def _finish_impl(self, elapsed_time, message):
        """Finish console progress bar."""
        # Fill the bar completely
        bar = "=" * self.width
        
        # Print final message with elapsed time
        sys.stdout.write(f"{self.title}: [{bar}] 100% - {message} ({elapsed_time:.2f}s elapsed)\n")
        sys.stdout.flush()


class QtProgressReporter(QObject, ProgressReporter):
    """Progress reporter that updates a Qt progress bar."""
    
    # Signal to update progress from any thread
    update_signal = pyqtSignal(int, str)
    finish_signal = pyqtSignal(float, str)
    
    def __init__(self, progress_bar=None, total_steps=100, title="Processing"):
        """
        Initialize Qt progress reporter.
        
        Parameters
        ----------
        progress_bar : QProgressBar, optional
            Progress bar widget to update
        total_steps : int
            Total number of steps in the operation
        title : str
            Title of the operation
        """
        QObject.__init__(self)
        ProgressReporter.__init__(self, total_steps, title)
        
        self.progress_bar = progress_bar
        
        # Connect signals
        self.update_signal.connect(self._update_progress_bar)
        self.finish_signal.connect(self._finish_progress_bar)
        
        # Initialize progress bar if provided
        if self.progress_bar:
            self.progress_bar.setRange(0, 100)
            self.progress_bar.setValue(0)
            if hasattr(self.progress_bar, 'setFormat'):
                self.progress_bar.setFormat(f"{self.title}: %p%")
    
    def _update_impl(self, percentage, message):
        """Emit signal to update progress bar."""
        self.update_signal.emit(int(percentage), message or "")
    
    def _update_progress_bar(self, percentage, message):
        """Update progress bar from main thread."""
        if self.progress_bar:
            self.progress_bar.setValue(percentage)
            
            # Update format if message provided
            if hasattr(self.progress_bar, 'setFormat') and message:
                self.progress_bar.setFormat(f"{self.title}: %p% - {message}")
            
            # Process events to ensure UI updates
            QApplication.processEvents(QEventLoop.ExcludeUserInputEvents)
    
    def _finish_impl(self, elapsed_time, message):
        """Emit signal to finish progress bar."""
        self.finish_signal.emit(elapsed_time, message)
    
    def _finish_progress_bar(self, elapsed_time, message):
        """Update progress bar for completion."""
        if self.progress_bar:
            self.progress_bar.setValue(100)
            
            if hasattr(self.progress_bar, 'setFormat'):
                self.progress_bar.setFormat(f"{self.title}: 100% - {message} ({elapsed_time:.2f}s)")
            
            # Process events to ensure UI updates
            QApplication.processEvents(QEventLoop.ExcludeUserInputEvents)


def create_progress_reporter(progress_bar=None, total_steps=100, title="Processing"):
    """
    Factory function to create appropriate progress reporter.
    
    Parameters
    ----------
    progress_bar : QProgressBar, optional
        Progress bar widget to update. If None, use console reporter
    total_steps : int
        Total number of steps in the operation
    title : str
        Title of the operation
        
    Returns
    -------
    ProgressReporter
        Appropriate progress reporter instance
    """
    if progress_bar is not None:
        return QtProgressReporter(progress_bar, total_steps, title)
    else:
        return ConsoleProgressReporter(total_steps, title)