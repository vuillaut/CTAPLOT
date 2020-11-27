import matplotlib.pyplot as plt

from .angular_resolution import AngularResolution
from .energy_resolution import EnergyResolution


class DL2:

    def __init__(self, angular_resolution, energy_resolution):

        self.angular_resolution = angular_resolution
        self.energy_resolution = energy_resolution

    def write(self, filename, **kwargs):
        self.angular_resolution.write(filename, **kwargs)
        kwargs.pop('overwrite')
        kwargs['append'] = True
        self.energy_resolution.write(filename, **kwargs)

    @classmethod
    def from_raw(cls, true_energy, reco_energy, true_alt, reco_alt, true_az, reco_az):
        ang_res = AngularResolution.from_raw(true_energy, true_alt, reco_alt, true_az, reco_az)
        ene_res = EnergyResolution.from_raw(true_energy, reco_energy)
        return cls(ang_res, ene_res)

    @classmethod
    def read(cls, filename):
        ang_res = AngularResolution.read(filename)
        ene_res = EnergyResolution.read(filename)
        return cls(ang_res, ene_res)

    def plot(self, axes=None, figsize=(12, 5), **kwargs):
        if axes is None:
            fig, axes = plt.subplots(1, 2, figsize=figsize)

        self.angular_resolution.plot(ax=axes[0], **kwargs)
        self.energy_resolution.plot(ax=axes[1], **kwargs)

        axes.ravel()[0].get_figure().tight_layout()
        return axes

