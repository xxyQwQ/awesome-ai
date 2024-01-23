import sys
import pickle


class Logger:
    def __init__(self, path):
        self.terminal = sys.stdout
        self.file = open(path, "w", encoding='utf-8')

    def write(self, message):
        self.terminal.write(message)
        self.file.write(message)
        self.file.flush()

    def flush(self):
        pass


class Recorder:
    def __init__(self, *entries):
        self.entries = entries
        self.records = {entry: [] for entry in entries}
    
    def load(self, path):
        with open(path, 'rb') as file:
            self.entries, self.records = pickle.load(file)
    
    def save(self, path):
        with open(path, 'wb') as file:
            pickle.dump((self.entries, self.records), file)
    
    def update(self, entry, value):
        if entry not in self.entries:
            self.entries.append(entry)
            self.records[entry] = []
        self.records[entry].append(value)
    
    def record(self, dictionary):
        for entry, value in dictionary.items():
            self.update(entry, value)
