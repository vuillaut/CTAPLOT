from .metrics import Metric
from .new_function import roc_gammaness_multiclass, plot_roc_multiclass
from astropy.table import Table
import numpy as np
from astropy.io.misc.hdf5 import write_table_hdf5
import os
import tables
from sklearn.metrics import auc


class ROCGamma(Metric):

    def __init__(self, data):
        """

        Parameters
        ----------
        data: dict with keys 'all' and each particle type different than gamma_label=0
            each value is a Table with the columns `fpr`, `tpr` and `thresholds`
        """
        if type(data) is not dict:
            raise TypeError("data must be a dict")
        self.data = data

    @classmethod
    def from_raw(cls, true_type, gammaness, gamma_label=0):
        rocs = roc_gammaness_multiclass(true_type, gammaness, gamma_label=gamma_label)
        data = {}
        for k, v in rocs.items():
            data[k] = Table(data=np.transpose(v),
                            names=['fpr', 'tpr', 'thresholds']
                            )
        return cls(data)

    def plot(self, ax=None, **kwargs):
        labels = {key: f'gamma vs {key} (auc={self.auc_score(key):.2f})' for key in self.data.keys()}
        ax = plot_roc_multiclass(self.data, ax=ax, labels=labels, **kwargs)
        ax.set_xlabel('gamma false positive rate')
        ax.set_ylabel('gamma true positive rate')
        return ax

    def write(self, filename, **kwargs):
        kwargs.setdefault('metadata_serialization', True)
        if 'overwrite' in kwargs and kwargs['overwrite'] and os.path.exists(filename):
            os.remove(filename)

        kwargs['overwrite'] = False
        kwargs['append'] = True
        if 'path' in kwargs:
            kwargs.pop('path')

        for key, table in self.data.items():
            write_table_hdf5(table, filename, path=f'classification/roc/{key}', **kwargs)

    @classmethod
    def read(cls, filename):
        data = {}
        with tables.open_file(filename) as file:
            node = file.root['classification/roc/']
            for key in node.__members__:
                data[key] = Table(node[key].read())

        return cls(data)

    def auc_score(self, key='all'):
        return auc(self.data[key]['fpr'], self.data[key]['tpr'])
