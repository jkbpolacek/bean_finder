# source: https://calmcode.io/
import logging
import tqdm

GLOBAL_LEVEL = logging.DEBUG
SHELL_LEVEL = logging.INFO
FILE_LEVEL = logging.DEBUG


class TqdmLoggingHandler(logging.Handler):
    # source: https://stackoverflow.com/questions/38543506/change-logging-print-function-to-tqdm-write-so-logging-doesnt-interfere-wit
    def __init__(self, level=logging.NOTSET):
        super().__init__(level)

    def emit(self, record):
        try:
            msg = self.format(record)
            tqdm.tqdm.write(msg)
            self.flush()
        except (KeyboardInterrupt, SystemExit):
            raise
        except:
            self.handleError(record)


logger = logging.getLogger(__name__)

shell_handler = TqdmLoggingHandler()
file_handler = logging.FileHandler("debug.log")

logger.setLevel(GLOBAL_LEVEL)
shell_handler.setLevel(SHELL_LEVEL)
file_handler.setLevel(FILE_LEVEL)

fmt_shell = "%(levelname)s %(asctime)s %(message)s"
fmt_file = (
    "%(levelname)s %(asctime)s [%(filename)s:%(funcName)s:%(lineno)d] %(message)s"
)

shell_formatter = logging.Formatter(fmt_shell)
file_formatter = logging.Formatter(fmt_file)

shell_handler.setFormatter(shell_formatter)
file_handler.setFormatter(file_formatter)

logger.addHandler(shell_handler)
logger.addHandler(file_handler)

# logger.debug('this is a debug statement')
# logger.info('this is a info statement')
# logger.warning('this is a warning statement')
# logger.critical('this is a critical statement')
# logger.error('this is a error statement')
