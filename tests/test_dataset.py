"""Unit tests for DatasetBuilder and BirdCLEFDataset (Task 7)."""

from __future__ import annotations

import csv
import os
import tempfile

import numpy as np
import pytest

from birdclef2026.src.dataset import (
    DatasetBuilder,
    BirdCLEFDataset,
    SampleRecord,
    NUM_CLASSES,
    _parse_secondary_labels,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _write_taxonomy(path: str, labels: list[str]) -> None:
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=["primary_label", "inat_taxon_id", "scientific_name", "common_name", "class_name"]
        )
        writer.writeheader()
        for lbl in labels:
            writer.writerow(
                {"primary_label": lbl, "inat_taxon_id": lbl,
                 "scientific_name": lbl, "common_name": lbl, "class_name": "Aves"}
            )


def _write_train_csv(path: str, rows: list[dict]) -> None:
    fieldnames = ["primary_label", "secondary_labels", "rating", "filename",
                  "type", "latitude", "longitude", "scientific_name",
                  "common_name", "class_name", "inat_taxon_id", "author",
                  "license", "url", "collection"]
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            full_row = {k: "" for k in fieldnames}
            full_row.update(row)
            writer.writerow(full_row)


def _make_taxonomy_labels(n: int = 234) -> list[str]:
    """Generate n unique label strings."""
    return [f"species{i:04d}" for i in range(n)]


def _make_builder(
    tmp_dir: str,
    labels: list[str] | None = None,
    train_rows: list[dict] | None = None,
    rating_threshold: float = 0.0,
) -> DatasetBuilder:
    if labels is None:
        labels = _make_taxonomy_labels(234)
    if train_rows is None:
        train_rows = [
            {"primary_label": labels[0], "secondary_labels": "[]",
             "rating": "5.0", "filename": "dummy.ogg"},
        ]

    tax_path = os.path.join(tmp_dir, "taxonomy.csv")
    train_path = os.path.join(tmp_dir, "train.csv")
    _write_taxonomy(tax_path, labels)
    _write_train_csv(train_path, train_rows)

    return DatasetBuilder(
        taxonomy_path=tax_path,
        train_csv_path=train_path,
        soundscape_labels_path=None,
        audio_dir=tmp_dir,
        soundscape_dir=None,
        rating_threshold=rating_threshold,
    )


# ---------------------------------------------------------------------------
# label_map tests
# ---------------------------------------------------------------------------

class TestLabelMap:
    def test_label_map_has_234_entries(self):
        with tempfile.TemporaryDirectory() as tmp:
            builder = _make_builder(tmp)
            assert len(builder.label_map) == NUM_CLASSES

    def test_label_map_values_are_unique_integers_0_to_233(self):
        with tempfile.TemporaryDirectory() as tmp:
            builder = _make_builder(tmp)
            values = sorted(builder.label_map.values())
            assert values == list(range(NUM_CLASSES))

    def test_label_map_is_consistent_across_instances(self):
        """Same taxonomy file always produces the same mapping."""
        with tempfile.TemporaryDirectory() as tmp:
            labels = _make_taxonomy_labels(234)
            tax_path = os.path.join(tmp, "taxonomy.csv")
            train_path = os.path.join(tmp, "train.csv")
            _write_taxonomy(tax_path, labels)
            _write_train_csv(train_path, [
                {"primary_label": labels[0], "secondary_labels": "[]",
                 "rating": "5.0", "filename": "dummy.ogg"}
            ])

            b1 = DatasetBuilder(tax_path, train_path, None, tmp, None)
            b2 = DatasetBuilder(tax_path, train_path, None, tmp, None)
            assert b1.label_map == b2.label_map

    def test_label_map_order_matches_taxonomy_row_order(self):
        """First label in taxonomy.csv gets index 0."""
        with tempfile.TemporaryDirectory() as tmp:
            labels = _make_taxonomy_labels(234)
            builder = _make_builder(tmp, labels=labels)
            assert builder.label_map[labels[0]] == 0
            assert builder.label_map[labels[1]] == 1
            assert builder.label_map[labels[233]] == 233


# ---------------------------------------------------------------------------
# Multi-hot label vector tests
# ---------------------------------------------------------------------------

class TestLabelVector:
    def test_label_vector_shape_and_dtype(self):
        with tempfile.TemporaryDirectory() as tmp:
            labels = _make_taxonomy_labels(234)
            rows = [{"primary_label": labels[5], "secondary_labels": "[]",
                     "rating": "5.0", "filename": "dummy.ogg"}]
            builder = _make_builder(tmp, labels=labels, train_rows=rows)
            samples = builder._load_train_audio_samples()
            vec = samples[0].label_vector
            assert vec.shape == (NUM_CLASSES,)
            assert vec.dtype == np.float32

    def test_primary_label_index_is_1(self):
        with tempfile.TemporaryDirectory() as tmp:
            labels = _make_taxonomy_labels(234)
            rows = [{"primary_label": labels[10], "secondary_labels": "[]",
                     "rating": "5.0", "filename": "dummy.ogg"}]
            builder = _make_builder(tmp, labels=labels, train_rows=rows)
            samples = builder._load_train_audio_samples()
            vec = samples[0].label_vector
            assert vec[10] == 1.0

    def test_secondary_labels_are_set(self):
        with tempfile.TemporaryDirectory() as tmp:
            labels = _make_taxonomy_labels(234)
            sec = f"['{labels[20]}', '{labels[30]}']"
            rows = [{"primary_label": labels[5], "secondary_labels": sec,
                     "rating": "5.0", "filename": "dummy.ogg"}]
            builder = _make_builder(tmp, labels=labels, train_rows=rows)
            samples = builder._load_train_audio_samples()
            vec = samples[0].label_vector
            assert vec[5] == 1.0
            assert vec[20] == pytest.approx(0.3)
            assert vec[30] == pytest.approx(0.3)

    def test_non_label_indices_are_zero(self):
        with tempfile.TemporaryDirectory() as tmp:
            labels = _make_taxonomy_labels(234)
            rows = [{"primary_label": labels[0], "secondary_labels": "[]",
                     "rating": "5.0", "filename": "dummy.ogg"}]
            builder = _make_builder(tmp, labels=labels, train_rows=rows)
            samples = builder._load_train_audio_samples()
            vec = samples[0].label_vector
            assert vec[0] == 1.0
            assert np.sum(vec) == 1.0  # only primary label set


# ---------------------------------------------------------------------------
# Rating threshold filtering tests
# ---------------------------------------------------------------------------

class TestRatingThreshold:
    def test_samples_below_threshold_are_excluded(self):
        with tempfile.TemporaryDirectory() as tmp:
            labels = _make_taxonomy_labels(234)
            rows = [
                {"primary_label": labels[0], "secondary_labels": "[]",
                 "rating": "1.0", "filename": "low.ogg"},
                {"primary_label": labels[1], "secondary_labels": "[]",
                 "rating": "4.0", "filename": "high.ogg"},
            ]
            builder = _make_builder(tmp, labels=labels, train_rows=rows, rating_threshold=3.0)
            samples = builder._load_train_audio_samples()
            assert len(samples) == 1
            assert "high.ogg" in samples[0].audio_path

    def test_samples_at_threshold_are_included(self):
        with tempfile.TemporaryDirectory() as tmp:
            labels = _make_taxonomy_labels(234)
            rows = [
                {"primary_label": labels[0], "secondary_labels": "[]",
                 "rating": "3.0", "filename": "exact.ogg"},
            ]
            builder = _make_builder(tmp, labels=labels, train_rows=rows, rating_threshold=3.0)
            samples = builder._load_train_audio_samples()
            assert len(samples) == 1

    def test_zero_threshold_includes_all(self):
        with tempfile.TemporaryDirectory() as tmp:
            labels = _make_taxonomy_labels(234)
            rows = [
                {"primary_label": labels[i], "secondary_labels": "[]",
                 "rating": str(float(i)), "filename": f"f{i}.ogg"}
                for i in range(5)
            ]
            builder = _make_builder(tmp, labels=labels, train_rows=rows, rating_threshold=0.0)
            samples = builder._load_train_audio_samples()
            assert len(samples) == 5


# ---------------------------------------------------------------------------
# Stratified split tests
# ---------------------------------------------------------------------------

class TestStratifiedSplit:
    def _make_samples(self, labels: list[str], label_map: dict[str, int],
                      counts: dict[str, int]) -> list[SampleRecord]:
        """Create dummy SampleRecord list with given per-label counts."""
        samples = []
        for lbl, n in counts.items():
            vec = np.zeros(NUM_CLASSES, dtype=np.float32)
            vec[label_map[lbl]] = 1.0
            for i in range(n):
                samples.append(SampleRecord(
                    audio_path=f"/fake/{lbl}_{i}.ogg",
                    start_sec=0.0,
                    end_sec=0.0,
                    label_vector=vec.copy(),
                    row_id=None,
                ))
        return samples

    def test_split_produces_both_sets(self):
        with tempfile.TemporaryDirectory() as tmp:
            labels = _make_taxonomy_labels(234)
            builder = _make_builder(tmp, labels=labels)
            counts = {labels[i]: 10 for i in range(5)}
            samples = self._make_samples(labels, builder.label_map, counts)
            train, val = builder._stratified_split(samples, val_fraction=0.2, seed=42)
            assert len(train) > 0
            assert len(val) > 0

    def test_split_covers_all_samples(self):
        with tempfile.TemporaryDirectory() as tmp:
            labels = _make_taxonomy_labels(234)
            builder = _make_builder(tmp, labels=labels)
            counts = {labels[i]: 10 for i in range(5)}
            samples = self._make_samples(labels, builder.label_map, counts)
            train, val = builder._stratified_split(samples, val_fraction=0.2, seed=42)
            assert len(train) + len(val) == len(samples)

    def test_labels_with_multiple_samples_appear_in_both_splits(self):
        """Every label with >= 2 samples should appear in both train and val."""
        with tempfile.TemporaryDirectory() as tmp:
            labels = _make_taxonomy_labels(234)
            builder = _make_builder(tmp, labels=labels)
            counts = {labels[i]: 5 for i in range(10)}
            samples = self._make_samples(labels, builder.label_map, counts)
            train, val = builder._stratified_split(samples, val_fraction=0.2, seed=42)

            def label_set(split):
                return {int(np.argmax(s.label_vector)) for s in split}

            train_labels = label_set(train)
            val_labels = label_set(val)
            for i in range(10):
                idx = builder.label_map[labels[i]]
                assert idx in train_labels, f"Label {labels[i]} missing from train"
                assert idx in val_labels, f"Label {labels[i]} missing from val"

    def test_single_sample_label_goes_to_train(self):
        """Labels with only 1 sample should go entirely to train."""
        with tempfile.TemporaryDirectory() as tmp:
            labels = _make_taxonomy_labels(234)
            builder = _make_builder(tmp, labels=labels)
            counts = {labels[0]: 1}
            samples = self._make_samples(labels, builder.label_map, counts)
            train, val = builder._stratified_split(samples, val_fraction=0.2, seed=42)
            assert len(train) == 1
            assert len(val) == 0

    def test_split_is_reproducible_with_same_seed(self):
        with tempfile.TemporaryDirectory() as tmp:
            labels = _make_taxonomy_labels(234)
            builder = _make_builder(tmp, labels=labels)
            counts = {labels[i]: 8 for i in range(5)}
            samples = self._make_samples(labels, builder.label_map, counts)
            train1, val1 = builder._stratified_split(samples, val_fraction=0.2, seed=42)
            train2, val2 = builder._stratified_split(samples, val_fraction=0.2, seed=42)
            assert [s.audio_path for s in train1] == [s.audio_path for s in train2]
            assert [s.audio_path for s in val1] == [s.audio_path for s in val2]


class TestGroupSplit:
    def test_group_split_keeps_group_ids_exclusive(self):
        with tempfile.TemporaryDirectory() as tmp:
            builder = _make_builder(tmp)
            samples = []
            for group_idx in range(5):
                for item_idx in range(3):
                    vec = np.zeros(NUM_CLASSES, dtype=np.float32)
                    vec[group_idx] = 1.0
                    samples.append(SampleRecord(
                        audio_path=f"/fake/g{group_idx}_{item_idx}.ogg",
                        start_sec=0.0,
                        end_sec=0.0,
                        label_vector=vec,
                        group_id=f"group-{group_idx}",
                    ))

            train, val = builder._group_split(samples, val_fraction=0.4, seed=42)
            train_groups = {s.group_id for s in train}
            val_groups = {s.group_id for s in val}

            assert train_groups.isdisjoint(val_groups)
            assert len(train) + len(val) == len(samples)
            assert len(val) > 0

    def test_soundscape_group_split_keeps_soundscape_files_exclusive(self):
        with tempfile.TemporaryDirectory() as tmp:
            builder = _make_builder(tmp)
            samples = []
            for group_idx in range(4):
                for item_idx in range(2):
                    vec = np.zeros(NUM_CLASSES, dtype=np.float32)
                    vec[group_idx] = 1.0
                    samples.append(SampleRecord(
                        audio_path=f"/fake/sc{group_idx}.ogg",
                        start_sec=item_idx * 5.0,
                        end_sec=(item_idx + 1) * 5.0,
                        label_vector=vec,
                        group_id=f"soundscape_file:sc{group_idx}",
                        source="soundscape",
                    ))

            train, val = builder._group_split(
                samples,
                val_fraction=0.5,
                seed=7,
                soundscape_only=True,
            )
            train_groups = {s.group_id for s in train if s.source == "soundscape"}
            val_groups = {s.group_id for s in val if s.source == "soundscape"}

            assert train_groups.isdisjoint(val_groups)


# ---------------------------------------------------------------------------
# Secondary label parsing
# ---------------------------------------------------------------------------

class TestParseSecondaryLabels:
    def test_empty_list_string(self):
        assert _parse_secondary_labels("[]") == []

    def test_python_list_literal(self):
        result = _parse_secondary_labels("['compau', 'saffin']")
        assert result == ["compau", "saffin"]

    def test_single_item_list(self):
        result = _parse_secondary_labels("['compau']")
        assert result == ["compau"]

    def test_space_separated(self):
        result = _parse_secondary_labels("compau saffin")
        assert result == ["compau", "saffin"]

    def test_empty_string(self):
        assert _parse_secondary_labels("") == []
