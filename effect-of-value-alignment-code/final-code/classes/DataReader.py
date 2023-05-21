import pickle

class PickleReader:

    def __init__(self, path) -> None:
        self.path = path
    
    def read_data(self):

        with open(self.path, 'rb') as f:
            self.data = pickle.load(f)
