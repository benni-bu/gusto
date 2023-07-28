"""Gusto Logging Module

All logging functionality for Gusto is controlled from
``gusto.logging``. A logger object ``logging.getLogger("gusto")`` is
created internally.

The primary means of configuration is via environment variables, the
same way that the standard Python root logger is. See the
:mod:`logging` page for details.

Set ``GUSTO_LOG_LEVEL`` to any of ``DEBUG``, ``INFO``, ``WARNING``,
``ERROR`` or ``CRITICAL`` (from most verbose to least verbose).

Additionally the level of console (`stderr`) logging and logfile based
logging can be controlled separately. Console logging verbosity is set
using ``GUSTO_CONSOLE_LOG_LEVEL``. Logfile logging verbosity is set using
``GUSTO_LOGFILE_LOG_LEVEL``.

By default a script that imports gusto will log only from rank 0 to the
console and to a file. This can be changed by setting the environment
variable ``GUSTO_PARALLEL_LOG`` to ``CONSOLE``, ``FILE`` or ``BOTH``.
Setting these will log from all ranks, not just rank 0. Console output
will be interleaved, but log files contain the rank number as part of
the logfile name.

"""

import logging
import sys
import os

from datetime import datetime
from logging import NOTSET, DEBUG, INFO, WARNING, ERROR, CRITICAL  # noqa: F401
from pathlib import Path

from pyop2.mpi import COMM_WORLD

__all__ = [
    "logger", "set_log_handler", "LoggingError", "NOTSET", "DEBUG",
    "INFO", "WARNING", "ERROR", "CRITICAL"
]


class LoggingError(Exception):
    pass


logging.captureWarnings(True)
logger = logging.getLogger("gusto")


def capture_exceptions(exception_type, exception_value, traceback, logger=logger):
    """ This function allows all unhandled exceptions to be logged to
    Gusto's logs
    """
    logger.error(
        "Gusto is logging this unhandled exception:",
        exc_info=(exception_type, exception_value, traceback)
    )


sys.excepthook = capture_exceptions

# Set the log level based on environment variables
log_level = os.environ.get("GUSTO_LOG_LEVEL", WARNING)
logfile_level = os.environ.get("GUSTO_FILE_LOG_LEVEL", DEBUG)
logconsole_level = os.environ.get("GUSTO_CONSOLE_LOG_LEVEL", INFO)
log_level_list = [log_level, logfile_level, logconsole_level]
log_levels = [
    logging.getLevelNamesMapping().get(x) if isinstance(x, str)
    else x
    for x in log_level_list
]
logger.setLevel(min(log_levels))

# Setup parallel logging based on environment variables
parallel_log = os.environ.get("GUSTO_PARALLEL_LOG", None)
options = ["CONSOLE", "FILE", "BOTH"]
if parallel_log is not None:
    parallel_log = parallel_log.upper()
    if parallel_log.upper() not in options:
        parallel_log = None


def create_logfile_handler(path, mode="w"):
    """ Handler for logfiles.

    Args:
        path: path to log file

    """
    logfile = logging.FileHandler(filename=path, mode=mode)
    logfile.setLevel(logfile_level)
    logfile_formatter = logging.Formatter(
        '%(asctime)s %(levelname)-8s %(message)s'
    )
    logfile.setFormatter(logfile_formatter)
    return logfile


def create_console_handler(fmt):
    """ Handler for console logging.

    Args:
        fmt: format string for log output
    """
    console = logging.StreamHandler()
    console.setLevel(logconsole_level)
    console_formatter = logging.Formatter(fmt)
    console.setFormatter(console_formatter)
    return console


def set_log_handler(comm=COMM_WORLD):
    """ Set all handlers for logging.

    Args:
        comm (:class:`MPI.Comm`): MPI communicator.
    """
    # Set up logging
    timestamp = datetime.now()
    logfile_name = f"temp-gusto-{timestamp.strftime('%Y-%m-%dT%H%M%S')}"
    if parallel_log in ["FILE", "BOTH"]:
        logfile_name += f"_{comm.rank}"
    logfile_name += ".log"
    if comm.rank == 0:
        os.makedirs("results", exist_ok=True)
    logfile_path = os.path.join("results", logfile_name)

    console_format_str = ""
    if parallel_log in ["CONSOLE", "BOTH"]:
        console_format_str += f"[{comm.rank}] "
    console_format_str += '%(levelname)-8s %(message)s'

    if comm.rank == 0:
        # Always log on rank 0
        lfh = create_logfile_handler(logfile_path)
        lfh.name = "gusto-temp-file-log"
        logger.addHandler(lfh)
        ch = create_console_handler(console_format_str)
        ch.name = "gusto-console-log"
        logger.addHandler(ch)
    else:
        # Only log on other ranks if enabled
        if parallel_log in ["FILE", "BOTH"]:
            lfh = create_logfile_handler(logfile_path)
            lfh.name = "gusto-temp-file-log"
            logger.addHandler(lfh)
        if parallel_log in ["CONSOLE", "BOTH"]:
            ch = create_console_handler(console_format_str)
            ch.name = "gusto-console-log"
            logger.addHandler(ch)
        if not logger.handlers:
            logger.addHandler(logging.NullHandler())

    logger.info("Running %s" % " ".join(sys.argv))


def update_logfile_location(new_path):
    """ Update the location of the logfile.

    This is used to move the temporary log file created in the results
    directory to the appropriate model directory.

    """
    new_path = Path(new_path)
    fh = [*filter(lambda x: x.name == "gusto-temp-file-log", logger.handlers)]

    if len(fh) == 1:
        fh = fh[0]
        logger.debug("Closing temporary logger and moving logfile")
        old_path = Path(fh.baseFilename)
        filename = Path(old_path.name.removeprefix("temp-"))
        fh.flush()
        fh.close()

        os.makedirs(new_path, exist_ok=True)
        os.rename(old_path, new_path/filename)

        new_fh = create_logfile_handler(new_path/filename, mode="a")
        new_fh.name = "gusto-file-log"
        logger.addHandler(new_fh)
        logger.debug("Re-opening logger")
    elif len(fh) > 1:
        raise LoggingError(
            "More than one log handler with name `gusto-temp-file-log`\n"
            "Logging has been set up incorrectly"
        )
