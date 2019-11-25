import logging.config

logging.config.fileConfig(r"util/log/conf/logger.conf")
logger = logging.getLogger("xiaoi")