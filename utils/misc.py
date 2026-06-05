import os
from contextlib import contextmanager

COLORS = [
    (33, 150, 243),  # Blue
    (244, 67, 54),   # Red
    (76, 175, 80),   # Green
    (255, 152, 0),   # Orange
    (121, 85, 72),   # Brown
    (158, 158, 158), # Grey
    (96, 125, 139),  # Blue Grey
    (233, 30, 99),   # Pink
    (0, 188, 212),   # Cyan
    (205, 220, 57),  # Lime
    (63, 81, 181),   # Indigo
    (139, 195, 74),  # Light Green
    (255, 193, 7),   # Amber
    (255, 87, 34),   # Deep Orange
    (103, 58, 183)   # Deep Purple
]

def check_key_and_bool(config: dict, key: str) -> bool:
    """Check the existance of the key and if it's True

    Args:
        config (dict): dict.
        key (str): Key name to be checked.

    Returns:
        bool: Return True only if the key exists in the dict and its value is True.
            Otherwise returns False.
    """
    return key in config.keys() and config[key]

def uniquify_dir(path):
    filename, extension = os.path.splitext(path)
    counter = 1
    while os.path.exists(path):
        path = f'{filename}-{counter}{extension}'
        counter += 1
    return path

@contextmanager
def suppress_stdout_stderr():
    """Context manager to suppress stdout and stderr"""
    # Save the current file descriptors
    save_stdout = os.dup(1)
    save_stderr = os.dup(2)
    null_fd = os.open(os.devnull, os.O_RDWR)

    # Redirect stdout and stderr to /dev/null
    os.dup2(null_fd, 1)
    os.dup2(null_fd, 2)

    try:
        yield
    finally:
        # Restore file descriptors
        os.dup2(save_stdout, 1)
        os.dup2(save_stderr, 2)
        # Clean up
        os.close(null_fd)
        os.close(save_stdout)
        os.close(save_stderr)