import logging
import random
import os
import uuid

from logging.handlers import TimedRotatingFileHandler

class UUIDFileHandler2(logging.FileHandler):
    def __init__(self, path, fileName, mode):
        fullpath = f"{path}/{fileName}-{uuid.uuid1()}"
        #os.mkdir(path)
        super(UUIDFileHandler2, self).__init__(fullpath,mode)

class UUIDFileHandler(TimedRotatingFileHandler):
    def __init__(self, path, fileName, mode):
        fullpath = f"{path}/{fileName}-{uuid.uuid1()}"
        super(UUIDFileHandler, self).__init__(filename=fullpath,
                                               when='midnight') # Restart the file every midnight. Use 'M' for testing on minutes