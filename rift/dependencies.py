"""
dependencies.py
===============
Central import Hub for the Rift harness. Every module in the rift/ package
pulls its third-party and standard-library dependencies from here.

Third-party packages:
    openai         - OpenAI-compatible client for NVIDIA NIM endpoints

Standard library:
    os, json, sys, time, threading, logging, subprocess, re, pathlib,
    datetime, typing
"""

# third-party
from openai import OpenAI

# standard library
import os
import json
import sys
import time
import threading
import logging
import subprocess
import re
from pathlib import Path
from datetime import datetime
from typing import Any, Optional

# shared constants
PACKAGE_DIR = Path(__file__).resolve().parent              # .../Rift/rift
BASE_DIR = PACKAGE_DIR.parent                              # .../Rift
MEMORY_FILE = BASE_DIR / "memory.md"
CONFIG_FILE = PACKAGE_DIR / "config.json"

DEFAULT_BASE_URL = "https://integrate.api.nvidia.com/v1"
DEFAULT_MODEL = "meta/llama-3.1-8b-instruct"

LOG_FILE = BASE_DIR / "rift.log"

# Define formatter for file logging
file_formatter = logging.Formatter(
    fmt="%(asctime)s  [%(levelname)s]  %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

# File handler (INFO and above)
file_handler = logging.FileHandler(LOG_FILE, mode="a")
file_handler.setFormatter(file_formatter)
file_handler.setLevel(logging.INFO)

# Console handler (clean, only for user-facing CLI messages)
console_handler = logging.StreamHandler(sys.stderr)
console_handler.setLevel(logging.INFO)

class ConsoleFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        msg = record.getMessage()
        if record.levelno >= logging.ERROR:
            return f"\033[1;31mError:\033[0m {msg}"
        elif record.levelno >= logging.WARNING:
            return f"\033[1;33mWarning:\033[0m {msg}"
        elif record.levelno == logging.INFO:
            return f"\033[90m{msg}\033[0m"  # dim grey
        return msg

console_handler.setFormatter(ConsoleFormatter())

# Configure root logger to output warnings/errors only (suppresses httpx/openai INFO logs)
root_logger = logging.getLogger()
root_logger.setLevel(logging.WARNING)
root_logger.addHandler(file_handler)

# Configure rift logger
log = logging.getLogger("rift")
log.setLevel(logging.INFO)
log.addHandler(file_handler)
log.addHandler(console_handler)
log.propagate = False
