"""
Code to generate interesting information
about the scBiGLasso algorithm.
"""

import numpy as np
import matplotlib.pyplot as plt
from Scripts.generate_data import *
from Scripts.utilities import *
from Scripts.scBiGLasso import *

def get_cms_for_betas(
    betas_to_try: "List of L1 penalties to try",
    attempts: "Amount of times we run the experiment to average over",
    kwargs_gen: "Dictionary of parameters for generating random data",
    kwargs_lasso: "Dictionary of parameters for the scBiGLasso algorithm",
    cm_mode: "`mode` argument for `generate_confusion_matrices`" = "Negative",
    verbose: bool = False
) -> (
    "List of all average confusion matrices for Psi",
    "List of all average confusion matrices for Theta",
):
    """
    We want to be able to make ROC curves parameterized by
    the L1 penalty.  This function will return confusion matrices
    to aid in that endeavor.
    
    We enforce beta_1 = beta_2.
    """
    
    Psi_cms = []
    Theta_cms = []
    for b in betas_to_try:
        if verbose:
            print(f"\n\nTrying beta={b:.6f}")
        Psi_cm = np.empty((attempts, 2, 2))
        Theta_cm = np.empty((attempts, 2, 2))
        for attempt in range(attempts):
            Psi_gen, Theta_gen, Ys = generate_Ys(**kwargs_gen)
            Psi, Theta = scBiGLasso(
                Ys=Ys,
                beta_1=b,
                beta_2=b,
                verbose=verbose,
                **kwargs_lasso
            )
            Psi_cm[attempt, ...] = generate_confusion_matrices(Psi, Psi_gen, mode=cm_mode)
            Theta_cm[attempt, ...] = generate_confusion_matrices(Theta, Theta_gen, mode=cm_mode)
        Psi_cms.append(Psi_cm.mean(axis=0))# / (Psi_cm.sum()))
        Theta_cms.append(Theta_cm.mean(axis=0))# / (Theta_cm.sum()))
        if verbose:
            print(f"\tPsi Confusion: \n{Psi_cms[-1]}")
            print(f"\tTheta Confusion: \n{Theta_cms[-1]}")
            print(f"\tPsi Precision: {precision(Psi_cms[-1])}")
            print(f"\tPsi Recall: {recall(Psi_cms[-1])}")
    return Psi_cms, Theta_cms

def make_cm_plots(
    betas: "List of L1 penalties",
    Psi_cms: "Corresponding confusion matrices for Psi",
    Theta_cms: "Corresponding confusion matrices for Theta",
    betas_to_highlight: "List of *indices* of beta values to highlight on graph" = [],
    title: "Title of the figure" = None
) -> ("Matplotlib Figure", "Tuple of Axes", ("Dict of Precisions", "Dict of Recalls")):
    
    precisions: "Keys are 'Psi'/'Theta'" = dict({})
    recalls:    "Keys are 'Psi'/'Theta'" = dict({})
    with plt.style.context('Solarize_Light2'):
        fig, (ax1, ax2) = plt.subplots(figsize=(16, 8), ncols=2)
        for confmats, ax, name in (
            (Psi_cms, ax1, "Psi"),
            (Theta_cms, ax2, "Theta")
        ):
            precisions[name] = [precision(cm) for cm in confmats]
            recalls[name] = [recall(cm) for cm in confmats]
            ax.plot(recalls[name], precisions[name])
            ax.set_xlabel("Recall")
            ax.set_ylabel("Precision")
            ax.set_title(name)
            ax.set_xlim([0, 1])
            ax.set_ylim([0, 1])
            for beta_to_highlight in betas_to_highlight:
                bval = betas[beta_to_highlight]
                cmval = confmats[beta_to_highlight]
                _precision = precision(cmval)
                _recall = recall(cmval)
                ax.annotate(
                    f'β={bval:.5f}',
                    xy=(_recall, _precision),
                    xytext=(_recall-0.1, _precision-0.1),
                    arrowprops=dict(facecolor='black', shrink=0.05)
                )
        if title is not None:
            fig.suptitle(title, fontsize=16)
        return fig, (ax1, ax2), (precisions, recalls)

def create_precision_recall_curves(
    betas_to_try: "List of L1 penalties to try",
    m: "Amount of samples",
    p: "Size of Psi/Theta",
    indices_to_highlight: "List of indices of betas to highlight on plot",
    attempts: "Number of times to average over" = 100,
    verbose: bool = False,
    expected_nonzero_psi = None,
    expected_nonzero_theta = None,
    lasso_every_loop = True,
):
    """
    Given a list of L1 penalties, calculate the 
    """
    n = p
    kwargs_gen = {
        'm': m,
        'p': p,
        'n': n,
        'structure': 'Kronecker Sum',
        'expected_nonzero_psi': n**2 / 5,
        'expected_nonzero_theta': p**2 / 5
    }
    if expected_nonzero_psi is not None:
        kwargs_gen['expected_nonzero_psi'] = expected_nonzero_psi
    else:
        expected_nonzero_psi = kwargs_gen['expected_nonzero_psi']
    if expected_nonzero_theta is not None:
        kwargs_gen['expected_nonzero_theta'] = expected_nonzero_theta
    else:
        expected_nonzero_theta = kwargs_gen['expected_nonzero_theta']
        
    kwargs_lasso = {
        'N': 100,
        'eps': 10e-3,
        'lasso_every_loop': lasso_every_loop
    }

    Psi_cms, Theta_cms = get_cms_for_betas(
        betas_to_try,
        attempts=attempts,
        kwargs_gen=kwargs_gen,
        kwargs_lasso=kwargs_lasso,
        verbose=verbose
    )
    
    psi_density = expected_nonzero_psi / (2*n*p)
    theta_density = expected_nonzero_theta / (2*n*p)
    
    return make_cm_plots(
        betas_to_try,
        Psi_cms,
        Theta_cms,
        indices_to_highlight,
        f"Precision-Recall Plots for {n}x{n} Psi/Theta as L1 Penalty β Varies "
        + f"({m} samples, {psi_density:.0%}/{theta_density:.0%} Psi/Theta densities)"
    )