import pickle


class PickleReader:
    """
    Reads a pickled data file
    """

    def __init__(self, path) -> None:
        """
        Initialize the reader with the path to the pickle file
        :param path: path to the pickled data file
        """
        self.data = None
        self.path = path

    def read_data(self):
        """
        Reads the data from the datafile
        """

        with open(self.path, 'rb') as f:
            self.data = pickle.load(f)
