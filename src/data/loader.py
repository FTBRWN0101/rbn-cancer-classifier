"""
load ovarian cancer gene expression data from arff
main entry is load_ovarian_data
"""
from __future__ import annotations
import json, re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np

@dataclass
class DatasetMetadata:
    """dataset metadata holder"""
    relation_name: str
    n_samples: int
    n_features: int
    n_classes: int
    feature_names: List[str]
    class_names: List[str]
    data_shape: Tuple[int, int]

    def to_dict(self) -> Dict:
        """convert metadata to dict"""
        return {
            "relation_name": self.relation_name,
            "n_samples": self.n_samples,
            "n_features": self.n_features,
            "n_classes": self.n_classes,
            "feature_names": self.feature_names,
            "class_names": self.class_names,
            "data_shape": self.data_shape,
        }

class ARFFLoader:
    """load and parse arff files"""
    _RELATION_PATTERN = re.compile(r"@RELATION\s+(\S+)", re.IGNORECASE)
    _ATTRIBUTE_PATTERN = re.compile(
        r"@ATTRIBUTE\s+([^\s]+)\s+(.+?)(?=\n@|\n\n|$)",
        re.IGNORECASE | re.DOTALL,
    )
    _DATA_PATTERN = re.compile(r"@DATA\s*\n(.+)", re.IGNORECASE | re.DOTALL)
    _CLASS_VALUES_PATTERN = re.compile(r"\{(.+?)\}")

    def __init__(self, filepath: str | Path) -> None:
        """init loader"""
        self.filepath = Path(filepath)
        if not self.filepath.exists():
            raise FileNotFoundError(f"ARFF file not found: {self.filepath}")
        self._relation_name: str = ""
        self._attributes: List[Tuple[str, str]] = []
        self._data: np.ndarray = np.array([])
        self._labels: np.ndarray = np.array([])
        self._feature_names: List[str] = []
        self._class_names: List[str] = []

    @property
    def data(self) -> np.ndarray:
        return self._data

    @property
    def labels(self) -> np.ndarray:
        return self._labels

    @property
    def feature_names(self) -> List[str]:
        return self._feature_names.copy()

    @property
    def metadata(self) -> DatasetMetadata:
        return DatasetMetadata(
            relation_name=self._relation_name,
            n_samples=len(self._data),
            n_features=len(self._feature_names),
            n_classes=len(self._class_names),
            feature_names=self._feature_names.copy(),
            class_names=self._class_names.copy(),
            data_shape=self._data.shape,
        )

    def load(self) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """
        load and parse the arff file

        Returns:
            data: np.ndarray (n_samples, n_features)
            labels: np.ndarray (n_samples,)
            feature_names: list[str]
        """
        content = self._read_file()
        self._parse_relation(content)
        self._parse_attributes(content)
        self._parse_data(content)
        self._validate_data()
        return self._data, self._labels, self.feature_names

    def save_numpy(self, output_dir: str | Path) -> None:
        """save data.npy labels.npy feature_names.txt metadata.json"""
        if self._data.size == 0:
            raise ValueError("No data loaded. Call load() first.")
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        np.save(output_dir / "data.npy", self._data)
        np.save(output_dir / "labels.npy", self._labels)
        with open(output_dir / "feature_names.txt", "w", encoding="utf-8") as f:
            f.write("\n".join(self._feature_names))
        with open(output_dir / "metadata.json", "w", encoding="utf-8") as f:
            json.dump(self.metadata.to_dict(), f, indent=2)

    def _read_file(self) -> str:
        with open(self.filepath, "r", encoding="utf-8") as f:
            return f.read()

    def _parse_relation(self, content: str) -> None:
        match = self._RELATION_PATTERN.search(content)
        if match:
            self._relation_name = match.group(1).strip("'\"")

    def _parse_attributes(self, content: str) -> None:
        matches = self._ATTRIBUTE_PATTERN.findall(content)
        for name, attr_type in matches:
            name = self._clean_attribute_name(name)
            attr_type = attr_type.strip()
            self._attributes.append((name, attr_type))
            if self._is_class_attribute(attr_type):
                self._parse_class_names(attr_type)
            else:
                self._feature_names.append(name)

    def _parse_data(self, content: str) -> None:
        match = self._DATA_PATTERN.search(content)
        if not match:
            raise ValueError("No @DATA section found in ARFF file")
        data_section = match.group(1).strip()
        lines = self._filter_data_lines(data_section)
        data_list, labels_list = self._parse_data_lines(lines)
        self._data = np.array(data_list, dtype=np.float64)
        self._labels = self._encode_labels(labels_list)

    def _validate_data(self) -> None:
        if self._data.size == 0:
            raise ValueError("No valid data found in ARFF file")
        if len(self._data) != len(self._labels):
            raise ValueError("Mismatch between data and labels length")
        if self._data.shape[1] != len(self._feature_names):
            raise ValueError("Mismatch between data columns and feature names")

    @staticmethod
    def _clean_attribute_name(name: str) -> str:
        return name.strip().strip("'\"")

    @staticmethod
    def _is_class_attribute(attr_type: str) -> bool:
        return "{" in attr_type and "}" in attr_type

    def _parse_class_names(self, attr_type: str) -> None:
        match = self._CLASS_VALUES_PATTERN.search(attr_type)
        if match:
            class_values = match.group(1)
            self._class_names = [c.strip().strip("'\"") for c in class_values.split(",")]

    @staticmethod
    def _filter_data_lines(data_section: str) -> List[str]:
        return [
            line.strip()
            for line in data_section.split("\n")
            if line.strip() and not line.strip().startswith("%")
        ]

    def _parse_data_lines(self, lines: List[str]) -> Tuple[List[np.ndarray], List[str]]:
        data_list: List[np.ndarray] = []
        labels_list: List[str] = []
        expected_length = len(self._attributes)
        for line_num, line in enumerate(lines, start=1):
            try:
                values = self._parse_line(line)
                if len(values) != expected_length:
                    continue
                features, label = values[:-1], values[-1]
                feature_array = np.array([float(f) for f in features])
                data_list.append(feature_array)
                labels_list.append(label)
            except (ValueError, IndexError) as e:
                print(f"Warning: Skipping malformed line {line_num}: {e}")
                continue
        return data_list, labels_list

    @staticmethod
    def _parse_line(line: str) -> List[str]:
        return [v.strip().strip("'\"") for v in line.split(",")]

    def _encode_labels(self, labels_list: List[str]) -> np.ndarray:
        unique_labels = sorted(set(labels_list))
        label_to_idx = {label: idx for idx, label in enumerate(unique_labels)}
        return np.array([label_to_idx[label] for label in labels_list])

def load_ovarian_data(filepath: str = "data/raw_data/Ovarian.arff") -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """
    convenience loader used by the pipeline

    Returns:
        data: np.ndarray (n_samples, n_features)
        labels: np.ndarray (n_samples,)
        feature_names: list[str]
    """
    loader = ARFFLoader(filepath)
    data, labels, feature_names = loader.load()
    print("Loaded ovarian cancer dataset:")
    print(f"  Samples: {data.shape[0]}")
    print(f"  Features: {data.shape[1]}")
    print(f"  Classes: {len(np.unique(labels))}")
    return data, labels, feature_names
