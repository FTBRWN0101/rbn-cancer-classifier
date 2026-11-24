"""Check the ARFF loader reads the file it is given and codes the labels alphabetically."""

import numpy as np

from data.loader import load_ovarian_data

ARFF = """@relation test

@attribute MZ1 numeric
@attribute MZ2 numeric
@attribute Class {Normal,Cancer}

@data

0.1,0.2,Normal
0.3,0.4,Cancer
0.5,0.6,Normal
"""


def test_loader_reads_given_path_and_codes_cancer_as_zero(tmp_path):
    path = tmp_path / "not_the_default.arff"
    path.write_text(ARFF, encoding="ascii")
    data, labels, names = load_ovarian_data(str(path))
    assert names == ["MZ1", "MZ2"]
    np.testing.assert_array_equal(data, [[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]])
    # Labels sort alphabetically, whatever order the header declares: Cancer is 0, Normal is 1.
    assert labels.tolist() == [1, 0, 1]
