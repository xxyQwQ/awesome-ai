import sys


class Logger:
    def __init__(self, path):
        self.terminal = sys.stdout
        self.file = open(path, "w")

    def write(self, message):
        self.terminal.write(message)
        self.file.write(message)
        self.file.flush()

    def flush(self):
        pass
