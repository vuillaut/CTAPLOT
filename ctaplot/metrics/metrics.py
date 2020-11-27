from astropy.io.misc.hdf5 import write_table_hdf5, read_table_hdf5
from astropy.table import Table, QTable

class Metric:
    """
    Generic metric class
    """

    def __init__(self, data):
        if not type(data) is Table or type(data) is QTable:
            raise TypeError("Data must be an astropy Table")
        self.data = data

    def write(self, filename, path=None, **kwargs):
        kwargs.setdefault('serialize_meta', True)
        write_table_hdf5(self.data, filename, path=path, **kwargs)

    @classmethod
    def read(cls, filename, path=None):
        data = read_table_hdf5(filename, path=path)
        return cls(data=data)



