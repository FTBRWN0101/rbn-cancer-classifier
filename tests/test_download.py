"""Check the download script's safeguards and its conversion, without touching the network."""

import download_data
import pytest

DBC = """// Data Base. Elvira Format

data-base ovarian_61902 {
number-of-cases = 2;

node SampleType(finite-states) {
kind-of-node = chance;
states = (Cancer Normal);
}

node MZ-7.86E-05 (continuous) {
kind-of-node = chance;
}

node MZ2.18E-07 (continuous) {
kind-of-node = chance;
}

relation  {

memory = true;

cases = (
[ Normal, 0.494626, 4.69E-4 ],
[ Cancer, 1.0,
0.0 ]
);

}
}
"""

ARFF = """@relation Orarian

@attribute MZ-7.86E-05 numeric
@attribute MZ2.18E-07 numeric
@attribute Class {Cancer,Normal}

@data

0.494626,0.000469,Normal
1,0,Cancer
"""


def test_dbc_converts_to_the_original_arff_layout():
    assert download_data.dbc_to_arff(DBC) == ARFF


def test_download_with_wrong_hash_is_refused_and_nothing_is_written(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(download_data, "fetch", lambda url: b"not the dataset")
    dest = tmp_path / "raw_data" / "Ovarian.arff"
    with pytest.raises(SystemExit):
        download_data.main(dest)
    assert not dest.parent.exists()


def test_converted_arff_with_wrong_hash_is_not_written(tmp_path):
    dest = tmp_path / "Ovarian.arff"
    with pytest.raises(SystemExit):
        download_data.write_checked(ARFF.encode("ascii"), dest)
    assert list(tmp_path.iterdir()) == []


def test_existing_file_that_fails_the_hash_is_left_untouched(tmp_path):
    dest = tmp_path / "Ovarian.arff"
    dest.write_text("someone else's file", encoding="ascii")
    with pytest.raises(SystemExit):
        download_data.main(dest)
    assert dest.read_text(encoding="ascii") == "someone else's file"


def test_plain_http_is_refused():
    with pytest.raises(ValueError, match="non-HTTPS"):
        download_data.fetch(
            "http://leo.ugr.es/elvira/DBCRepository/OvarianCancer/Ovarian-PBSII-061902.zip"
        )
