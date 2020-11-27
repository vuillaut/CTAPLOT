import astropy.units as u
import numpy as np
from astropy.table import Table

from .metrics import Metric
from .. import ana
from .new_function import plot_angular_resolution


class AngularResolution(Metric):

    def __init__(self, data):
        super().__init__(data=data)

    @classmethod
    def from_raw(cls, true_energy, true_alt, reco_alt, true_az, reco_az):
        e_bins, ang_res = ana.angular_resolution_per_energy(reco_alt.to_value(u.rad),
                                                            reco_az.to_value(u.rad),
                                                            true_alt.to_value(u.rad),
                                                            true_az.to_value(u.rad),
                                                            true_energy.to_value(u.TeV))
        e_min = e_bins[:-1] * u.TeV
        e_max = e_bins[1:] * u.TeV
        angular_resolution = ang_res[:, 0] * u.rad
        angular_resolution_err = np.array([ang_res[:, 0] - ang_res[:, 1], ang_res[:, 2] - ang_res[:, 0]]) * u.rad

        data = Table(data=np.vstack([e_min, e_max]).T,
                     names=['e_min', 'e_max']
                     )
        data['angular_resolution'] = angular_resolution
        data['angular_resolution_err'] = angular_resolution_err.T

        return cls(data)

    def plot(self, ax=None, **kwargs):
        ax = plot_angular_resolution(self.data['e_min'],
                                     self.data['e_max'],
                                     self.data['angular_resolution'],
                                     self.data['angular_resolution_err'].T,
                                     ax=ax,
                                     **kwargs,
                                     )
        return ax

    def write(self, filename, **kwargs):
        super().write(filename, path='angular_resolution', **kwargs)

    @classmethod
    def read(cls, filename):
        return super().read(filename, path='angular_resolution')
