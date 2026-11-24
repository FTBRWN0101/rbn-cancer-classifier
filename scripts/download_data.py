"""Download the ovarian cancer serum proteomics data and write data/raw_data/Ovarian.arff.

Fetches the Kent Ridge copy of the NCI 6-19-02 ovarian dataset from the Elvira project at
leo.ugr.es, falling back to the Internet Archive's copy of the same file.
Checks the zip and the ARFF it writes against pinned SHA-256 hashes, and keeps nothing that fails.
Uses the standard library only.
"""

from __future__ import annotations

import hashlib
import io
import os
import re
import tempfile
import urllib.request
import zipfile
from decimal import Decimal
from pathlib import Path

DEST = Path(__file__).resolve().parent.parent / "data" / "raw_data" / "Ovarian.arff"
URLS = (
    "https://leo.ugr.es/elvira/DBCRepository/OvarianCancer/Ovarian-PBSII-061902.zip",
    "https://web.archive.org/web/20110805044034id_/http://leo.ugr.es/elvira/DBCRepository/OvarianCancer/Ovarian-PBSII-061902.zip",
)
ZIP_SHA256 = "f74a7a64e7ecbe279762d9ed4b9941d332253606b1776ec10a48f3ca1f46e452"
ZIP_BYTES = 35_718_363
ARFF_SHA256 = "277c8ffb77c7366fe15ea51f89879cd1efcaa9dedc13d0ae12ec743f39936858"
# The zip's other member is a serialised Java object; never read it.
MEMBER = "ovarian_61902.dbc"
CITATION = """\
Data: Petricoin, E. F. et al. (2002). Use of proteomic patterns in serum to identify ovarian cancer.
The Lancet 359(9306): 572-577. https://doi.org/10.1016/S0140-6736(02)07746-2
Copy: Kent Ridge Bio-medical Data Set Repository, via the Elvira project (leo.ugr.es)."""


def sha256(data: bytes) -> str:
    """Return the SHA-256 hex digest of data."""
    return hashlib.sha256(data).hexdigest()


def fetch(url: str, limit: int = ZIP_BYTES) -> bytes:
    """Return the body of an HTTPS URL, refusing anything larger than limit bytes."""
    if not url.startswith("https://"):
        raise ValueError(f"refusing a non-HTTPS URL: {url}")
    request = urllib.request.Request(
        url, headers={"User-Agent": "rbn-cancer-classifier"}
    )
    with urllib.request.urlopen(request, timeout=120) as response:
        if not response.geturl().startswith("https://"):
            raise ValueError(
                f"refusing a redirect to a non-HTTPS URL: {response.geturl()}"
            )
        body = response.read(limit + 1)
    if len(body) > limit:
        raise ValueError(f"the download is larger than the expected {limit} bytes")
    return body


def download_zip() -> bytes:
    """Return the dataset zip from the first source whose copy matches the pinned hash."""
    for url in URLS:
        try:
            body = fetch(url)
        except (OSError, ValueError) as err:
            print(f"  {url}: {err}")
            continue
        if sha256(body) == ZIP_SHA256:
            print(f"  {url}: hash matches")
            return body
        print(f"  {url}: hash does not match, so this copy is ignored")
    raise SystemExit("No source returned the expected file. Nothing was written.")


def format_value(token: str) -> str:
    """Write a .dbc number as the original ARFF does: fixed point, without trailing zeros."""
    text = format(Decimal(token), "f")
    return text.rstrip("0").rstrip(".") if "." in text else text


def dbc_to_arff(dbc: str) -> str:
    """Convert the Elvira .dbc text to the ARFF layout the original run read."""
    names = re.findall(r"^node (\S+) \(continuous\)", dbc, flags=re.MULTILINE)
    states = (
        re.search(r"^states = \(([^)]*)\);", dbc, flags=re.MULTILINE).group(1).split()
    )
    cases = re.findall(r"\[([^\]]*)\]", dbc[dbc.index("cases = (") :])
    # The relation name, typo included, is the original file's, so the output matches its hash.
    lines = ["@relation Orarian", ""]
    lines += [f"@attribute {name} numeric" for name in names]
    lines += [f"@attribute Class {{{','.join(states)}}}", "", "@data", ""]
    for case in cases:
        label, *values = (v.strip() for v in case.split(","))
        if label not in states or len(values) != len(names):
            raise ValueError(f"malformed case starting {case[:40]!r}")
        lines.append(",".join([*map(format_value, values), label]))
    return "\n".join(lines) + "\n"


def write_checked(data: bytes, dest: Path) -> None:
    """Write data to dest only if it matches the pinned ARFF hash."""
    if sha256(data) != ARFF_SHA256:
        raise SystemExit(
            "The converted ARFF does not match the pinned hash. Nothing was written."
        )
    dest.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=dest.parent, suffix=".part")
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(data)
        os.replace(tmp, dest)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise


def main(dest: Path = DEST) -> None:
    """Download, convert and check the dataset, leaving any existing copy untouched."""
    if dest.exists():
        if sha256(dest.read_bytes()) != ARFF_SHA256:
            raise SystemExit(
                f"{dest} exists but does not match the pinned hash. It was left untouched; move it away and rerun."
            )
        print(f"{dest} is already present and matches the pinned hash.")
    else:
        print("Downloading the ovarian cancer dataset (36 MB)")
        with zipfile.ZipFile(io.BytesIO(download_zip())) as archive:
            dbc = archive.read(MEMBER).decode("ascii")
        write_checked(dbc_to_arff(dbc).encode("ascii"), dest)
        print(f"Wrote {dest}")
    print(CITATION)


if __name__ == "__main__":
    main()
