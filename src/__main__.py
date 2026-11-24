# src/__main__.py
import sys

from data.runit import main as data_runit
from evaluation.runit import main as evaluation_runit
from feature_selection.runit import main as feature_selection_runit
from nk_landscape.runit import main as nk_landscape_runit
from rbn.runit import main as rbn_runit
from utils.runit import main as utils_runit

def main():
    # The stages print check marks, which Windows cannot encode when output goes to a file or pipe.
    sys.stdout.reconfigure(encoding="utf-8")
    print("\nRunning data handling scripts")
    data_runit()
    print("\nRunning feature selection scripts")
    feature_selection_runit()
    print("\nRunning evaluation scripts")
    evaluation_runit()
    print("\nRunning NK landscape scripts")
    nk_landscape_runit()
    print("\nRunning RBN scripts")
    rbn_runit()
    print("\nRunning utility scripts")
    utils_runit()

if __name__ == "__main__":
    main()
