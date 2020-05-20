import logging
import re


class InfoFilter(logging.Filter):
    def filter(self, record):
        if record.levelno >= logging.WARNING:
            return False
        return True


class LossFilter(logging.Filter):
    def filter(self, record):
        if re.findall("loss", record.getMessage()):
            return True
        return False


class logger():

    def __init__(self, modelname, filename, filemode, mode="INFO"):

        if mode == "INFO":
            log_level = logging.INFO
        elif mode == "DEBUG":
            log_level = logging.DEBUG
        elif model == "WARN":
            log_level = logging.WARNING
        elif model == "ERROR":
            log_level = logging.ERROR
        else:
            log_level = logging.CRITICAL

        logging.basicConfig(
            level=log_level, \
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', \
            datefmt="%Y-%m-%d %H:%M:%S", \
            filename=filename, \
            filemode=filemode
        )
        self.log = logging.getLogger("tensorflow")

        self.log.info("***************************Starting train {}********************************".format(modelname))

    def addInfoFilter(self):
        self.log.addFilter(InfoFilter())

    def addLossFilter(self):
        self.log.addFilter(LossFilter())




