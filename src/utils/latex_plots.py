import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
mpl.use("pgf")
pgf_with_pdflatex = {
    "pgf.texsystem": "pdflatex",
    "pgf.preamble": "\n".join([
        r"\usepackage[utf8x]{inputenc}",
        r"\usepackage[T1]{fontenc}",
        r"\usepackage{cmbright}",
    ])
}
mpl.rcParams.update(pgf_with_pdflatex)
def plot_curve_latex(x: np.ndarray, y: np.ndarray, title: str, filename: str):
    plt.figure(figsize=(4, 3))
    plt.plot(x, y, label='Data')
    plt.title(title)
    plt.xlabel('Step')
    plt.ylabel('Value')
    plt.legend()
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()
