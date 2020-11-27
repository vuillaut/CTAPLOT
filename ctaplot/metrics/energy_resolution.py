import numpy as np
import astropy.units as u
from astropy.table import Table

from ..ana import energy_resolution_per_energy

from .new_function import plot_energy_resolution
from .metrics import Metric



class EnergyResolution(Metric):

    def __init__(self, data):
        super().__init__(data=data)

    @classmethod
    def from_raw(cls, true_energy, reco_energy):
        E, ene_res = energy_resolution_per_energy(true_energy, reco_energy)
        e_min = E[:-1] * u.TeV
        e_max = E[1:] * u.TeV
        energy_resolution = ene_res[:, 0]
        energy_resolution_err = np.array([ene_res[:, 0] - ene_res[:, 1],
                                          ene_res[:, 2] - ene_res[:, 0],
                                          ]
                                         )

        data = Table(data=np.vstack([e_min, e_max]).T,
                     names=['e_min', 'e_max']
                     )
        data['energy_resolution'] = energy_resolution
        data['energy_resolution_err'] = energy_resolution_err.T
        return cls(data=data)

    def plot(self, ax=None, **kwargs):
        ax = plot_energy_resolution(self.data['e_min'],
                                    self.data['e_max'],
                                    self.data['energy_resolution'],
                                    self.data['energy_resolution_err'].T,
                                    ax=ax,
                                    **kwargs,
                                    )
        return ax

    def write(self, filename, **kwargs):
        super().write(filename, path='energy_resolution', **kwargs)

    @classmethod
    def read(self, filename):
        return super().read(filename, path='energy_resolution')
