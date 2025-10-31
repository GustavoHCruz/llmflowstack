from enum import Enum


class LogLevel(str, Enum):
  INFO = "info"
  WARNING = "warning"
  ERROR = "error"
  DEBUG = "debug"