"""
Code to generate interesting information
about the scBiGLasso algorithm.
"""

import numpy as np
import matplotlib.pyplot as plt
from Scripts.generate_data import *
from Scripts.utilities import *
from Scripts.scBiGLasso import *
from Scripts.anBiGLasso import *
from Scripts.EiGLasso import *
from Scripts.TeraLasso import *
from Scripts.anBiGLasso_cov import anBiGLasso as anBiGLasso_cov
from Scripts.antGLasso import antGLasso, antGLasso_heuristic
from Scripts.nonparanormal_skeptic import *
from cycler import cycler
from itertools import product

def get_cms_for_betas(
    betas_to_try: "List of L1 penalties to try",
    attempts: "Amount of times we run the experiment to average over",
    kwargs_gen: "Dictionary of parameters for generating random data",
    kwargs_lasso: "Dictionary of parameters for the scBiGLasso algorithm",
    cm_mode: "`mode` argument for `generate_confusion_matrices`" = "Negative",
    verbose: bool = False,
    alg: str = "scBiGLasso"
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
        Psi_cm = np.empty((2, 2))
        Theta_cm = np.empty((2, 2))
        for attempt in range(attempts):
            (Psi_gen, Theta_gen), Ys = generate_Ys(**kwargs_gen)
            if alg == "scBiGLasso":
                Psi, Theta = scBiGLasso(
                    Ys=Ys,
                    beta_1=b,
                    beta_2=b,
                    Psi_init=None,#np.eye(Ys.shape[-1]),
                    Theta_init=None,#np.eye(Ys.shape[-1]),
                    verbose=verbose,
                    **kwargs_lasso
                )
            elif alg == "anBiGLasso":
                Psi, Theta = anBiGLasso(
                    Ys=Ys,
                    beta_1=b,
                    beta_2=b,
                    **kwargs_lasso
                )
            elif alg == "anBiGLasso_cov":
                T, S = calculate_empirical_covariance_matrices(Ys)
                Psi, Theta = anBiGLasso_cov(
                    T=T,
                    S=S,
                    beta_1=b,
                    beta_2=b,
                    **kwargs_lasso
                )
            elif alg == "antGLasso_heuristic":
                T, S = calculate_empirical_covariance_matrices(Ys)
                Psi, Theta = antGLasso_heuristic(
                    [T, S],
                    beta_1=b,
                    beta_2=b,
                    **kwargs_lasso
                )
            elif alg == "anBiGLasso_cov_with_skeptic":
                T, S = nonparanormal_skeptic(Ys)
                Psi, Theta = anBiGLasso_cov(
                    T=T,
                    S=S,
                    beta_1=b,
                    beta_2=b,
                    **kwargs_lasso
                )
            elif alg == "EiGLasso":
                Psi, Theta = EiGLasso(
                    Ys=Ys,
                    beta_1=b,
                    beta_2=b,
                    **kwargs_lasso
                )
            else:
                raise ValueError(f"no such algorithm {alg}")
            Psi_cm += generate_confusion_matrices(Psi, Psi_gen, mode=cm_mode)
            Theta_cm += generate_confusion_matrices(Theta, Theta_gen, mode=cm_mode)
        Psi_cms.append(Psi_cm / (Psi_cm.sum()))
        Theta_cms.append(Theta_cm / (Theta_cm.sum()))
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
    alg: str = "scBiGLasso",
    df_scale: "int >= 1" = 1,
    B_approx_iters: int = 10,
    cm_mode = "Negative"
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
        'expected_nonzero_psi': p**2 / 5,
        'expected_nonzero_theta': n**2 / 5,
        'df_scale': df_scale
    }
    if alg == "scBiGLasso":
        kwargs_lasso = {
            "N": 100,
            "eps": 10e-3,
        }
    elif (
        alg == "anBiGLasso"
        or alg == "anBiGLasso_cov"
        or alg == "anBiGLasso_cov_with_skeptic"
        or alg == "antGLasso_heuristic"
    ):
        kwargs_lasso = {
            "B_approx_iters": B_approx_iters
        }
    elif alg == "EiGLasso":
        kwargs_lasso = dict({})
    else:
        raise ValueError(f"no such algorithm {alg}")

    Psi_cms, Theta_cms = get_cms_for_betas(
        betas_to_try,
        attempts=attempts,
        kwargs_gen=kwargs_gen,
        kwargs_lasso=kwargs_lasso,
        verbose=verbose,
        alg=alg,
        cm_mode=cm_mode
    )
    
    return make_cm_plots(
        betas_to_try,
        Psi_cms,
        Theta_cms,
        indices_to_highlight,
        f"Precision-Recall Plots for {n}x{n} Psi/Theta as L1 Penalty β Varies ({m} samples)"
    )

def get_cms_for_betas_all_algs(
    betas_to_try: "List of L1 penalties to try",
    attempts: "Amount of times we run the experiment to average over",
    kwargs_gen: "Dictionary of parameters for generating random data",
    cm_mode: "`mode` argument for `generate_confusion_matrices`" = "Negative",
    algorithms = None,
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
    
    if algorithms is None:
        algorithms = ["scBiGLasso", "anBiGLasso", "antGLasso_heuristic", "EiGLasso", "TeraLasso"]
    
    Psi_cms = np.zeros((*betas_to_try.shape, 2, 2))
    Theta_cms = np.zeros((*betas_to_try.shape, 2, 2))
    for idx_alg, alg in enumerate(algorithms):
        if verbose:
            print(f"Trying algorithm: {alg}")
        for idx_b, b in enumerate(betas_to_try[idx_alg]):
            if verbose:
                print(f"\tTrying beta={b:.6f}")
            for attempt in range(attempts):
                (Psi_gen, Theta_gen), Ys = generate_Ys(**kwargs_gen)
                if alg == "scBiGLasso":
                    Psi, Theta = scBiGLasso(
                        Ys=Ys,
                        beta_1=b,
                        beta_2=b,
                        Psi_init=None,
                        Theta_init=None,
                        N=100,
                        eps=10e-3
                    )
                elif alg == "anBiGLasso":
                    Psi, Theta = anBiGLasso(
                        Ys=Ys,
                        beta_1=b,
                        beta_2=b,
                        B_approx_iters=1000
                    )
                elif alg == "antGLasso":
                    Psis = antGLasso(
                        Ys=Ys,
                        betas=[b, b],
                        B_approx_iters=1000
                    )
                    Psi = Psis[0]
                    Theta = Psis[1]
                elif alg == "Hungry anBiGLasso":
                    Psi, Theta = anBiGLasso(
                        Ys=Ys,
                        beta_1=b,
                        beta_2=b,
                        B_approx_iters=-1
                    )
                elif alg == "anBiGLasso_cov":
                    T, S = calculate_empirical_covariance_matrices(Ys)
                    Psi, Theta = anBiGLasso_cov(
                        T=T,
                        S=S,
                        beta_1=b,
                        beta_2=b,
                        B_approx_iters=10
                    )
                elif alg == "antGLasso_heuristic":
                    T, S = calculate_empirical_covariance_matrices(Ys)
                    Psi, Theta = antGLasso_heuristic(
                        [T, S],
                        betas=[b, b],
                        B_approx_iters=10
                    )
                elif alg == "EiGLasso":
                    Psi, Theta = EiGLasso(
                        Ys=Ys,
                        beta_1=b,
                        beta_2=b
                    )
                elif alg == "TeraLasso":
                    Psi, Theta = TeraLasso(
                        Ys,
                        [b, b]
                    )
                else:
                    raise ValueError(f"no such algorithm {alg}")
                Psi_cms[idx_alg, idx_b, ...] += generate_confusion_matrices(
                    Psi,
                    Psi_gen,
                    mode=cm_mode
                ) / attempts
                Theta_cms[idx_alg, idx_b, ...] += generate_confusion_matrices(
                    Theta,
                    Theta_gen,
                    mode=cm_mode
                ) / attempts
                # End of attempts loop
            # End of betas loop
        # End of algorithms loop
    return Psi_cms, Theta_cms

def make_cm_plots_all_algs(
    Psi_cms: "Corresponding confusion matrices for Psi",
    Theta_cms: "Corresponding confusion matrices for Theta",
    algorithms = None,
    title = None
) -> ("Matplotlib Figure", "Tuple of Axes"):
    if algorithms is None:
        algorithms = ["scBiGLasso", "anBiGLasso", "antGLasso_heuristic", "EiGLasso", "TeraLasso"]
    with plt.style.context('Solarize_Light2'):
        plt.rcParams['axes.prop_cycle'] = cycler(color=[
            '#006BA4',
            '#FF800E',
            '#ABABAB',
            '#595959',
            '#5F9ED1',
            '#C85200',
            '#898989',
            '#A2C8EC',
            '#FFBC79',
            '#CFCFCF'
        ])
        fig, (ax1, ax2) = plt.subplots(figsize=(16, 8), ncols=2)
        for confmats, ax, name in (
            (Psi_cms, ax1, "Psi"),
            (Theta_cms, ax2, "Theta")
        ):
            precisions = dict({})
            recalls = dict({})
            for idx_alg, alg in enumerate(algorithms):
                precisions[name] = [precision(cm) for cm in confmats[idx_alg, ...]]
                recalls[name] = [recall(cm) for cm in confmats[idx_alg, ...]]
                ax.plot(recalls[name], precisions[name], label=alg)
                ax.set_xlabel("Recall")
                ax.set_ylabel("Precision")
                ax.set_title(name)
                ax.set_xlim([0, 1])
                ax.set_ylim([0, 1])
                ax.legend()
        if title is not None:
            fig.suptitle(title, fontsize=16)
        return fig, (ax1, ax2)
    
    
def create_precision_recall_curves_all(
    betas_to_try: "Tensor of L1 penalties to try",
    m: "Amount of samples",
    p: "Size of Psi/Theta",
    attempts: "Number of times to average over" = 100,
    verbose: bool = False,
    algorithms: list = None,
    df_scale: "int >= 1" = 1,
    cm_mode = "Negative",
    title = None
):
    """
    Given a list of L1 penalties, calculate the 
    """
    kwargs_gen = {
        'm': m,
        'ds': [p, p],
        'expected_nonzero': p**2 / 5,
        'df_scale': df_scale
    }

    Psi_cms, Theta_cms = get_cms_for_betas_all_algs(
        betas_to_try,
        attempts=attempts,
        kwargs_gen=kwargs_gen,
        algorithms=algorithms,
        verbose=verbose,
        cm_mode=cm_mode
    )
    
    data_out = np.empty((*betas_to_try.shape, 2, 2, 2))
    data_out[:, :, 0, :, :] = Psi_cms
    data_out[:, :, 1, :, :] = Theta_cms
    
    return *make_cm_plots_all_algs(
        Psi_cms,
        Theta_cms,
        algorithms=algorithms,
        title=title
    ), data_out

def get_cms_for_betas_tensor(
    betas_to_try: "List of L1 penalties to try",
    attempts: "Amount of times we run the experiment to average over",
    kwargs_gen: "Dictionary of parameters for generating random data",
    cm_mode: "`mode` argument for `generate_confusion_matrices`" = "Negative",
    algorithms = None,
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
    
    ds = kwargs_gen['ds']
    
    if algorithms is None:
        algorithms = ["antGLasso", "TeraLasso"]
    
    Psis_cms = np.zeros((*betas_to_try.shape, len(ds), 2, 2))
    for idx_alg, alg in enumerate(algorithms):
        if verbose:
            print(f"Trying algorithm: {alg}")
        for idx_b, b in enumerate(betas_to_try[idx_alg]):
            if verbose:
                print(f"\tTrying beta={b:.6f}")
            for attempt in range(attempts):
                Psis_gen, Ys = generate_Ys(**kwargs_gen)
                if alg == "antGLasso":
                    Psis = antGLasso(
                        Ys=Ys,
                        betas=[b for _ in ds],
                        B_approx_iters=10
                    )
                elif alg == "TeraLasso":
                    Psis = TeraLasso(
                        Ys,
                        [b for _ in ds]
                    )
                else:
                    raise ValueError(f"no such algorithm {alg}")
                Psis_cms[idx_alg, idx_b, ...] += np.array([generate_confusion_matrices(
                    Psis[idx],
                    Psis_gen[idx],
                    mode=cm_mode
                ) for idx in range(len(ds))]) / attempts
                # End of attempts loop
            # End of betas loop
        # End of algorithms loop
    return Psis_cms

def create_precision_recall_curves_tensor(
    betas_to_try: "Tensor of L1 penalties to try",
    m: "Amount of samples",
    ds: "List of sizes of precision matrices",
    attempts: "Number of times to average over" = 100,
    verbose: bool = False,
    algorithms: list = None,
    df_scale: "int >= 1" = 1,
    cm_mode = "Negative",
    title = None
):
    """
    Given a list of L1 penalties, calculate the 
    """
    kwargs_gen = {
        'm': m,
        'ds': ds,
        'expected_nonzero': ds[0]**2 / 5,
        'df_scale': df_scale
    }

    Psis_cms = get_cms_for_betas_tensor(
        betas_to_try,
        attempts=attempts,
        kwargs_gen=kwargs_gen,
        algorithms=algorithms,
        verbose=verbose,
        cm_mode=cm_mode
    )
    
    return *make_cm_plots_tensor(
        Psis_cms,
        algorithms=algorithms,
        title=title
    ), Psis_cms

def make_cm_plots_tensor(
    Psis_cms: "List of corresponding confusion matrices for each Psi",
    algorithms = None,
    title = None
) -> ("Matplotlib Figure", "Tuple of Axes"):
    D = Psis_cms.shape[2]
    if algorithms is None:
        algorithms = ["antGLasso", "TeraLasso"]
    with plt.style.context('Solarize_Light2'):
        plt.rcParams['axes.prop_cycle'] = cycler(color=[
            '#006BA4',
            '#FF800E',
            '#ABABAB',
            '#595959',
            '#5F9ED1',
            '#C85200',
            '#898989',
            '#A2C8EC',
            '#FFBC79',
            '#CFCFCF'
        ])
        fig, axs = plt.subplots(figsize=(8, 8*D), nrows=D)
        for idx, (confmats, ax) in enumerate(zip(np.rollaxis(Psis_cms, 2), axs)):
            name = f"Psi{idx}"
            precisions = dict({})
            recalls = dict({})
            for idx_alg, alg in enumerate(algorithms):
                precisions[name] = [precision(cm) for cm in confmats[idx_alg, ...]]
                recalls[name] = [recall(cm) for cm in confmats[idx_alg, ...]]
                ax.plot(recalls[name], precisions[name], label=alg)
                ax.set_xlabel("Recall")
                ax.set_ylabel("Precision")
                ax.set_title(name)
                ax.set_xlim([0, 1])
                ax.set_ylim([0, 1])
                ax.legend()
        if title is not None:
            fig.suptitle(title, fontsize=16)
        return fig, axs
    
def kwargs_generator(sizes, samples, df_scale=2, sparsity=0.2):
    yield from ({
        'm': m,
        'ds': ds,
        'expected_nonzero': int(ds[0]**2 * sparsity),
        'df_scale': df_scale
    }
        for ds, m in product(sizes, samples)
    )
    
def get_cms_for_betas_antGLasso(
    betas_to_try: "List of L1 penalties to try",
    attempts: "Amount of times we run the experiment to average over",
    cm_mode: "`mode` argument for `generate_confusion_matrices`",
    sizes: "List of problem sizes to try",
    samples: "List of problem samples to try",
    verbose: bool = False,
    df_scale: int = 2,
    try_sparsities: bool = False,
    sparsity: "0 <= x <= 1" = 0.2
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

    Psis_cms = np.zeros((*betas_to_try.shape, 2, 2, 2))
    for idx_alg, kwargs_gen in enumerate(kwargs_generator(sizes, samples, df_scale, sparsity)):
        for idx_b, b in enumerate(betas_to_try[idx_alg]):
            if verbose:
                print(f"\tTrying beta={b:.6f}")
            for attempt in range(attempts):
                Psis_gen, Ys = generate_Ys(**kwargs_gen)
                regularizer = {'betas': [b, b]} if not try_sparsities else {'sparsities': [b, b]}
                Psis = antGLasso(
                    Ys=Ys,
                    B_approx_iters=10,
                    **regularizer
                )
                Psis_cms[idx_alg, idx_b, ...] += np.array([generate_confusion_matrices(
                    Psis[idx],
                    Psis_gen[idx],
                    mode=cm_mode
                ) for idx in range(2)]) / attempts
                # End of attempts loop
            # End of betas loop
        # End of algorithms loop
    return Psis_cms

def make_cm_plots_antGLasso(
    Psis_cms: "List of corresponding confusion matrices for each Psi",
    sizes: "List of problem sizes to try",
    samples: "List of problem samples to try",
    title = None,
    betas_to_highlight: "List of *indices* of beta values to highlight on graph" = None,
    betas = None
) -> ("Matplotlib Figure", "Tuple of Axes"):
    D = Psis_cms.shape[2]
    with plt.style.context('Solarize_Light2'):
        plt.rcParams['axes.prop_cycle'] = cycler(color=[
            '#006BA4',
            '#FF800E',
            '#ABABAB',
            '#595959',
            '#5F9ED1',
            '#C85200',
            '#898989',
            '#A2C8EC',
            '#FFBC79',
            '#CFCFCF'
        ])
        fig, axs = plt.subplots(figsize=(8, 8*D), nrows=D)
        for idx, (confmats, ax) in enumerate(zip(np.rollaxis(Psis_cms, 2), axs)):
            name = f"Psi{idx}"
            precisions = dict({})
            recalls = dict({})
            for idx_alg, (size, sample) in enumerate(product(sizes, samples)):
                precisions[name] = [precision(cm) for cm in confmats[idx_alg, ...]]
                recalls[name] = [recall(cm) for cm in confmats[idx_alg, ...]]
                ax.plot(recalls[name], precisions[name], label=f"{size=} {sample=}")
                ax.set_xlabel("Recall")
                ax.set_ylabel("Precision")
                ax.set_title(name)
                ax.set_xlim([0, 1])
                ax.set_ylim([0, 1])
                ax.legend()
                if betas_to_highlight is not None:
                    for beta_to_highlight in betas_to_highlight[idx_alg]:
                        bval = betas[idx_alg, beta_to_highlight]
                        cmval = confmats[idx_alg, beta_to_highlight]
                        _precision = precision(cmval)
                        _recall = recall(cmval)
                        ax.annotate(
                            f'keep top {100*bval:.0f}%',
                            xy=(_recall, _precision),
                            xytext=(_recall-0.1, _precision-0.1),
                            arrowprops=dict(facecolor='black', shrink=0.05)
                        )
        if title is not None:
            fig.suptitle(title, fontsize=16)
        return fig, axs