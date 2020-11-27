

#### THIS SHOULD GO IN ANA ####
import numpy as np

from sklearn.metrics import roc_curve, roc_auc_score

def bin_arithmetic_mean(x_min, x_max):
    return (x_min + x_max)/2.

def bin_geometric_mean(x_min, x_max):
    return np.sqrt(x_min * x_max)


def roc_multiclass(true_type, reco_proba, pos_label, sample_weight=None, drop_intermediate=True):
    """

    Parameters
    ----------
    true_type
    reco_proba: `dict` of `numpy.ndarray` of shape `(len(simu_type), len(set(simu_type))`
        reconstruction probability for each class in `simu_type`, values must be between 0 and 1
    pos_label
    sample_weight
    drop_intermediate

    Returns
    -------
    """

    if pos_label not in set(true_type) or pos_label not in reco_proba:
        raise ValueError(f"simu_type and reco_proba must contain pos_label {pos_label}")

    for label in reco_proba.keys():
        if label not in true_type:
            raise ValueError("reco_proba must include all the labels in true_type")

    data = {}
    for label in reco_proba.keys():
        data[label] = roc_curve(true_type, reco_proba[label], pos_label=label,
                                sample_weight=sample_weight,
                                drop_intermediate=drop_intermediate,
                                )

    return data


def roc_gammaness(true_type, gammaness, gamma_label=0, sample_weight=None, drop_intermediate=True):

    fpr, tpr, thresholds = roc_curve(true_type, gammaness,
                                          pos_label=gamma_label,
                                          sample_weight=sample_weight,
                                          drop_intermediate=drop_intermediate,
                                          )

    return fpr, tpr, thresholds

def roc_gammaness_multiclass(true_type, gammaness, gamma_label=0, sample_weight=None, drop_intermediate=True):
    """

    Parameters
    ----------
    true_type
    gammaness
    gamma_label
    sample_weight
    drop_intermediate

    Returns
    -------

    """

    if gamma_label not in true_type:
        raise ValueError(f"gamma label {gamma_label} not in true_type")

    data = {}
    data['all'] = roc_gammaness(true_type, gammaness,
                                gamma_label=gamma_label,
                                sample_weight=sample_weight,
                                drop_intermediate=drop_intermediate,
                                )
    if len(set(true_type)) == 2:
        return data
    else:
        all_types = set(true_type)
        all_types.remove(gamma_label)
        for type in all_types:
            data[type] = roc_gammaness(true_type[(true_type == type) | (true_type == gamma_label)],
                                       gammaness[(true_type == type) | (true_type == gamma_label)],
                                       gamma_label=gamma_label,
                                       sample_weight=sample_weight,
                                       drop_intermediate=drop_intermediate
                                       )

    return data



###################################


### THIS GOES IN PLOT ###

import matplotlib.pyplot as plt
from astropy.visualization import quantity_support
import astropy.units as u
from matplotlib import ticker


def plot_resolution(x_min, x_max, resolution, resolution_err=None, log=False, ax=None, **kwargs):
    """
    Plot a resolution with errobars

    Parameters
    ----------
    x_min: `numpy.ndarray` or `astropy.quantity`
    x_max: `numpy.ndarray` or `astropy.quantity`
    resolution: `numpy.ndarray` or `astropy.quantity`
    resolution_err: `numpy.ndarray` or `astropy.quantity`
    log: bool
    ax: `matplotlib.pyplot.axis`
    kwargs: kwargs for `matplotlib.pyplot.errorbar`

    Returns
    -------
    `matplotlib.pyplot.axis`
    """
    ax = plt.gca() if ax is None else ax

    if not log:
        x = bin_arithmetic_mean(x_min, x_max)
    else:
        x = bin_geometric_mean(x_min, x_max)
        ax.set_xscale('log')

    if 'fmt' not in kwargs:
        kwargs['fmt'] = 'o'

    with quantity_support():
        ax.errorbar(x,
                    resolution,
                    xerr=[x - x_min, x_max - x],
                    yerr=resolution_err,
                    **kwargs
                    )

    ax.set_title('Resolution')
    return ax


def plot_angular_resolution(e_min, e_max, angular_resolution, angular_resolution_err=None, ax=None, **kwargs):
    energy_unit = u.Quantity(e_min).unit

    ax = plot_resolution(e_min,
                         e_max,
                         angular_resolution.to(u.deg),
                         angular_resolution_err.to(u.deg),
                         log=True,
                         ax=ax,
                         **kwargs,
                         )

    ax.set_title('Angular resolution')

    ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%0.2f'))

    ax.set_xlabel(r'{name} [{unit}]'.format(name='$E_{true}$', unit=energy_unit.to_string('latex')))
    ax.set_ylabel(r'{name} [{unit}]'.format(name='$PSF_{R68}$', unit=u.deg))
    ax.grid(True, which='both')
    return ax



def plot_energy_resolution(e_min, e_max, energy_resolution, energy_resolution_err=None, ax=None, **kwargs):
    energy_unit = u.Quantity(e_min).unit

    ax = plot_resolution(e_min, e_max, energy_resolution, energy_resolution_err, log=True, ax=ax, **kwargs)
    ax.set_title('Energy resolution')

    ax.set_xlabel(r'{name} [{unit}]'.format(name='$E_{true}$', unit=energy_unit.to_string('latex')))
    ax.set_ylabel(r'$\left( \frac{|E_{true} - E_{reco}|}{E_{true}} \right)_{R68}$')
    ax.grid(True, which='both')
    return ax


def plot_roc_multiclass(roc_dict, ax=None, labels=None, **kwargs):
    """

    Parameters
    ----------
    roc_dict: dict
    ax: `matplotlib.pyplot as plt`
    labels: dict of labels
        override 'label' in kwargs
    kwargs: kwargs for `matplotlib.pyplot.plot`

    Returns
    -------

    """
    ax = plt.gca() if ax is None else ax

    if labels is None:
        if "label" in kwargs:
            labels = {
                k: f"{kwargs['label']} {k}" for k in roc_dict.keys()
            }
        else:
            labels = {k: str(k) for k in roc_dict.keys()}

    for k, v in roc_dict.items():
        kwargs['label'] = labels[k]
        ax.plot(v['fpr'], v['tpr'], **kwargs)

    ax.plot([0, 1], [0, 1], ls='--', color='black')

    ax.set_xlabel('false positive rate')
    ax.set_ylabel('true positive rate')

    ax.legend()
    ax.grid(True)
    ax.axis('equal')
    return ax
