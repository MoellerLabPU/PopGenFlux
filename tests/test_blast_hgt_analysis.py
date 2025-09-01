#!/usr/bin/env python3
"""
HGT analysis test suite for `blast_hgt_analysis.py`.

This module contains comprehensive tests for the BLAST HGT analysis pipeline,
covering multiple analysis scenarios including within_mouse, within_replicate,
between_replicates, and global summaries.

What this tests:
- Sequence ID mapping utilities (`_map_seqids_to_mag_ids`)
- Pair canonicalization utilities (`_canonical_pair`)
- End-to-end processing on simple and complex synthetic datasets
- Statistical validation of sums, means, and percentages across all analysis levels
- Count-based statistical tests (binomial/poisson)
- Distributional tests for context-level comparisons

Test Design:
-----------
- Simple dataset: 2 replicates, 2 SGBs each, validates basic numerator/denominator logic
- Complex dataset: 4 replicates, 3+ samples each, validates aggregation and mean calculations
- Both datasets cover within_mouse, within_replicate, between_replicates, and global levels
- All tests validate expected values against known dataset characteristics

Run Instructions:
----------------
From the repository root:
- Preferred: `python3 -m pytest tests/test_blast_hgt_analysis.py`
- Specific test: `python3 -m pytest tests/test_blast_hgt_analysis.py::test_name`
- Verbose: `python3 -m pytest tests/test_blast_hgt_analysis.py -v`

Prerequisites:
-------------
- Python 3.9+
- pytest
- numpy, pandas, scipy, matplotlib

Test Coverage:
-------------
✅ Sequence ID mapping validity
✅ Pair canonicalization correctness
✅ Hit classification accuracy
✅ Denominator calculations across all levels
✅ Summary statistics (sums, means, percentages)
✅ Statistical test validation
✅ Edge cases and error handling
"""
import os
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import popgenflux.blast_hgt_analysis as bha


def _build_contig_map_df(mag_ids):
    """Construct a minimal contig→MAG map consistent with test BLAST IDs.

    Tests use gene IDs like 'MAG_X.fa_contig_1'. The mapping requires the contig
    prefix before the last underscore, i.e., 'MAG_X.fa_contig'.
    """
    return pd.DataFrame(
        {
            "mag_id": list(mag_ids),
            "contig_id": [f"{m}.fa_contig" for m in mag_ids],
        }
    )


def test_map_seqids_to_mag_ids_success():
    contig_map_df = _build_contig_map_df(["MAG_A", "MAG_B"])
    seqids = pd.Series(["MAG_A.fa_contig_1", "MAG_B.fa_contig_7"])  # contigID_geneID
    out = bha._map_seqids_to_mag_ids(seqids, contig_map_df)
    assert list(out) == ["MAG_A", "MAG_B"]


def test_map_seqids_to_mag_ids_raises_on_unmapped():
    contig_map_df = _build_contig_map_df(["MAG_A"])  # no mapping for MAG_X
    seqids = pd.Series(["MAG_A.fa_contig_1", "MAG_X.fa_contig_9"])  # second should fail
    with pytest.raises(SystemExit):
        _ = bha._map_seqids_to_mag_ids(seqids, contig_map_df)


def test_canonical_pair():
    """Canonicalization should make unordered pairs comparable.

    We rely on this to de-duplicate A–B vs B–A BLAST hits and to define
    unique MAG-pair keys.
    """
    a = pd.Series(["b", "a", "c"])  # a vs b/c
    b = pd.Series(["a", "b", "c"])  # b vs a/c (last is self)
    first, second = bha._canonical_pair(a, b)
    assert list(first) == ["a", "a", "c"]
    assert list(second) == ["b", "b", "c"]


def _build_mapping_df():
    """Minimal mapping for a simple scenario used by the basic test.

    Six MAGs across two replicates, two SGBs (X and Y). This dataset is crafted
    so each analysis level (within mouse, within replicate, between replicates,
    global) has at least one non-zero case and a known denominator.
    """
    # Six MAGs across two replicates, two SGBs (X and Y)
    rows = [
        {
            "MAG_ID": "MAG_A",
            "SGB": "SGB_X",
            "sample_id": "s1",
            "replicate": "R1",
            "subjectID": "sub1",
            "time": "t1",
            "group": "g1",
        },
        {
            "MAG_ID": "MAG_B",
            "SGB": "SGB_Y",
            "sample_id": "s1",
            "replicate": "R1",
            "subjectID": "sub1",
            "time": "t1",
            "group": "g1",
        },
        {
            "MAG_ID": "MAG_C",
            "SGB": "SGB_X",
            "sample_id": "s2",
            "replicate": "R1",
            "subjectID": "sub2",
            "time": "t1",
            "group": "g1",
        },
        {
            "MAG_ID": "MAG_D",
            "SGB": "SGB_Y",
            "sample_id": "s2",
            "replicate": "R1",
            "subjectID": "sub2",
            "time": "t1",
            "group": "g1",
        },
        {
            "MAG_ID": "MAG_E",
            "SGB": "SGB_X",
            "sample_id": "s3",
            "replicate": "R2",
            "subjectID": "sub3",
            "time": "t1",
            "group": "g1",
        },
        {
            "MAG_ID": "MAG_F",
            "SGB": "SGB_Y",
            "sample_id": "s4",
            "replicate": "R2",
            "subjectID": "sub4",
            "time": "t1",
            "group": "g1",
        },
    ]
    return pd.DataFrame(rows)


def _write_blast_file(tmp_path: Path) -> Path:
    """Create a toy BLAST file covering each analysis level.

    Notes on rows:
    - A vs B and B vs A are provided to ensure reverse duplicates are dropped
      after canonicalization.
    - A vs A is included to verify self-hits are removed.
    - All rows pass the strict filters (pident=100, evalue=0, length>=500).
    Columns: qseqid sseqid pident length mismatch gapopen qstart qend sstart send evalue bitscore
    """
    # Columns: qseqid sseqid pident length mismatch gapopen qstart qend sstart send evalue bitscore
    rows = [
        [
            "MAG_A.fa_contig_1",
            "MAG_B.fa_contig_7",
            100.0,
            600,
            0,
            0,
            1,
            600,
            100,
            700,
            0.0,
            1000,
        ],  # within_mouse (s1)
        [
            "MAG_B.fa_contig_7",
            "MAG_A.fa_contig_1",
            100.0,
            600,
            0,
            0,
            1,
            600,
            100,
            700,
            0.0,
            1000,
        ],  # duplicate reverse (should drop)
        [
            "MAG_A.fa_contig_1",
            "MAG_D.fa_contig_3",
            100.0,
            600,
            0,
            0,
            1,
            600,
            10,
            610,
            0.0,
            900,
        ],  # within_replicate (R1, s1 vs s2)
        [
            "MAG_A.fa_contig_1",
            "MAG_F.fa_contig_11",
            100.0,
            600,
            0,
            0,
            1,
            600,
            20,
            620,
            0.0,
            800,
        ],  # between_replicates (R1 vs R2)
        [
            "MAG_A.fa_contig_1",
            "MAG_A.fa_contig_1",
            100.0,
            600,
            0,
            0,
            1,
            600,
            1,
            600,
            0.0,
            1200,
        ],  # self (should drop)
    ]
    df = pd.DataFrame(rows)
    blast_path = tmp_path / "blast.tsv"
    df.to_csv(blast_path, sep="\t", header=False, index=False)
    return blast_path


def test_process_blast_file_and_summaries(tmp_path):
    """End-to-end check on the simple dataset.

    Verifies:
    - self-hits and reversed duplicates are filtered
    - classification across the three levels is correct
    - numerators/denominators for global and stratified summaries are as expected
    """
    map_df = _build_mapping_df()
    blast_path = _write_blast_file(tmp_path)

    contig_map_df = _build_contig_map_df(
        ["MAG_A", "MAG_B", "MAG_C", "MAG_D", "MAG_E", "MAG_F"]
    )
    results, inter_df = bha.process_blast_file(
        blast_path=blast_path,
        map_df=map_df,
        group="g1",
        timepoint="t1",
        pident=100.0,
        evalue=0.0,
        length=500,
        contig_map_df=contig_map_df,
    )

    # Inter-SGB results should contain 3 unique canonical MAG pairs
    assert not inter_df.empty
    assert inter_df["canonical_pair_key"].nunique() == 3
    assert set(inter_df["hgt_category"].unique()) == {
        "within_mouse",
        "within_replicate",
        "between_replicates",
    }

    # Global detailed expectations
    global_det = results["global"]["detailed_summary"]
    assert global_det.empty

    # Global aggregated expectations
    global_agg = results["global"]["aggregated_summary"]
    assert not global_agg.empty
    row = global_agg[
        (global_agg["SGB_MAG1"] == "SGB_X") & (global_agg["SGB_MAG2"] == "SGB_Y")
    ].iloc[0]
    assert int(row["sum_interacting_mag_pairs"]) == 3  # A-B, A-D, A-F
    assert int(row["sum_total_possible_mag_pairs"]) == 9  # 3 X by 3 Y
    assert np.isclose(
        (row["sum_interacting_mag_pairs"] / row["sum_total_possible_mag_pairs"]) * 100,
        (3 / 9) * 100,
    )

    # Within mouse (s1)
    within_mouse_det = results["within_mouse"]["detailed_summary"]
    wm_row = within_mouse_det[(within_mouse_det["context_id"] == "s1")].iloc[0]
    assert int(wm_row["interacting_mag_pairs"]) == 1
    assert int(wm_row["total_number_of_HGTs"]) == 1
    assert int(wm_row["total_possible_mag_pairs"]) == 1
    assert np.isclose(wm_row["percentage_interacting"], 100.0)

    # Within mouse aggregated summary for SGB_X and SGB_Y
    within_mouse_agg = results["within_mouse"]["aggregated_summary"]
    wm_agg_row = within_mouse_agg[
        (within_mouse_agg["SGB_MAG1"] == "SGB_X")
        & (within_mouse_agg["SGB_MAG2"] == "SGB_Y")
    ].iloc[0]
    assert int(wm_agg_row["sum_interacting_mag_pairs"]) == 1
    assert int(wm_agg_row["sum_total_number_of_HGTs"]) == 1
    assert int(wm_agg_row["sum_total_possible_mag_pairs"]) == 2
    assert np.isclose(wm_agg_row["mean_interacting_mag_pairs"], 0.25)  # 1/4
    assert np.isclose(wm_agg_row["mean_total_number_of_HGTs"], 0.25)  # 1/4
    assert np.isclose(wm_agg_row["mean_total_possible_mag_pairs"], 0.5)  # 2/4

    # Within replicate detailed summary for R1 and SGB_X, SGB_Y
    within_rep_det = results["within_replicate"]["detailed_summary"]
    wr_det_row = within_rep_det[
        (within_rep_det["context_id"] == "R1")
        & (within_rep_det["SGB_MAG1"] == "SGB_X")
        & (within_rep_det["SGB_MAG2"] == "SGB_Y")
    ].iloc[0]
    assert int(wr_det_row["interacting_mag_pairs"]) == 1  # A-D hit
    assert int(wr_det_row["total_number_of_HGTs"]) == 1
    assert int(wr_det_row["total_possible_mag_pairs"]) == 2  # (1*1)+(1*1)
    assert np.isclose(wr_det_row["percentage_interacting"], 50.0)

    # Within replicate R1 aggregated
    # Denominator logic within replicate: for each pair of mice in a replicate,
    # count X of mouse1 × Y of mouse2 PLUS X of mouse2 × Y of mouse1.
    # For s1 (X=1,Y=1) and s2 (X=1,Y=1) this is (1×1)+(1×1)=2.
    within_rep_agg = results["within_replicate"]["aggregated_summary"]
    wr_row = within_rep_agg[
        (within_rep_agg["SGB_MAG1"] == "SGB_X")
        & (within_rep_agg["SGB_MAG2"] == "SGB_Y")
    ].iloc[0]
    assert int(wr_row["sum_interacting_mag_pairs"]) == 1
    # Aggregated across R1 and R2: denom R1=2 (s1-s2), R2=1 (s3-s4) => total 3
    assert int(wr_row["sum_total_possible_mag_pairs"]) == 3
    # Mean is over contexts (R1 and R2): (1 + 0)/2 = 0.5
    assert np.isclose(wr_row["mean_interacting_mag_pairs"], 0.5)
    assert np.isclose(wr_row["mean_total_possible_mag_pairs"], 1.5)
    assert np.isclose(
        (wr_row["sum_interacting_mag_pairs"] / wr_row["sum_total_possible_mag_pairs"])
        * 100,
        (1 / 3) * 100,
    )

    # Between replicates detailed summary for (R1,R2) and SGB_X, SGB_Y
    between_rep_det = results["between_replicates"]["detailed_summary"]
    br_det_row = between_rep_det[
        (between_rep_det["context_id"].astype(str) == "('R1', 'R2')")
        & (between_rep_det["SGB_MAG1"] == "SGB_X")
        & (between_rep_det["SGB_MAG2"] == "SGB_Y")
    ].iloc[0]
    assert int(br_det_row["interacting_mag_pairs"]) == 1  # A-F hit
    assert int(br_det_row["total_number_of_HGTs"]) == 1
    assert int(br_det_row["total_possible_mag_pairs"]) == 4  # (2*1)+(1*2)
    assert np.isclose(br_det_row["percentage_interacting"], 25.0)

    # Between replicates aggregated
    between_rep_agg = results["between_replicates"]["aggregated_summary"]
    br_row = between_rep_agg[
        (between_rep_agg["SGB_MAG1"] == "SGB_X")
        & (between_rep_agg["SGB_MAG2"] == "SGB_Y")
    ].iloc[0]
    assert int(br_row["sum_interacting_mag_pairs"]) == 1
    assert int(br_row["sum_total_possible_mag_pairs"]) == 4  # (2*1)+(1*2)
    assert (
        int(br_row["mean_total_possible_mag_pairs"]) == 4
    )  # there is only a single context (R1,R2), mean equals the value
    assert np.isclose(br_row["mean_interacting_mag_pairs"], 1.0)
    assert np.isclose(
        (br_row["sum_interacting_mag_pairs"] / br_row["sum_total_possible_mag_pairs"])
        * 100,
        (1 / 4) * 100,
    )


def _build_very_complex_mapping_df() -> pd.DataFrame:
    """Very complex dataset: >=4 replicates, >=3 samples each, some samples with >3 MAGs and >3 SGBs.

    Replicates R1..R4; samples are R{rep}_S{idx}.
    SGBs include SGB_A, SGB_B, SGB_C, SGB_D. Some samples contain >3 SGB categories and >3 MAGs.
    """

    def rows_for(replicate, sample, sgb_to_mag_counts, start_idx):
        rows = []
        counters = {k: 0 for k in sgb_to_mag_counts}
        idx = start_idx
        for sgb, count in sgb_to_mag_counts.items():
            for _ in range(count):
                counters[sgb] += 1
                mag_id = f"{sgb.split('_')[-1]}{idx}"
                rows.append(
                    {
                        "MAG_ID": mag_id,
                        "SGB": sgb,
                        "sample_id": f"{replicate}_{sample}",
                        "replicate": replicate,
                        "subjectID": f"sub_{replicate}_{sample}",
                        "time": "t1",
                        "group": "g1",
                    }
                )
                idx += 1
        return rows, idx

    rows = []
    idx = 1
    # R1: S1 has >3 MAGs and >3 SGBs
    r, idx = rows_for("R1", "S1", {"SGB_A": 2, "SGB_B": 2, "SGB_C": 1, "SGB_D": 1}, idx)
    rows += r
    r, idx = rows_for("R1", "S2", {"SGB_A": 1, "SGB_B": 1, "SGB_C": 1}, idx)
    rows += r
    r, idx = rows_for("R1", "S3", {"SGB_A": 2, "SGB_C": 1, "SGB_D": 1}, idx)
    rows += r
    # R2
    r, idx = rows_for("R2", "S1", {"SGB_A": 1, "SGB_B": 2, "SGB_C": 1}, idx)
    rows += r
    r, idx = rows_for("R2", "S2", {"SGB_A": 1, "SGB_B": 1, "SGB_D": 1}, idx)
    rows += r
    r, idx = rows_for("R2", "S3", {"SGB_A": 2, "SGB_B": 1, "SGB_C": 1}, idx)
    rows += r
    # R3
    r, idx = rows_for("R3", "S1", {"SGB_A": 1, "SGB_B": 1, "SGB_C": 1}, idx)
    rows += r
    r, idx = rows_for("R3", "S2", {"SGB_A": 2, "SGB_B": 2}, idx)
    rows += r
    r, idx = rows_for("R3", "S3", {"SGB_A": 1, "SGB_C": 2, "SGB_D": 1}, idx)
    rows += r
    # R4
    r, idx = rows_for("R4", "S1", {"SGB_B": 2, "SGB_C": 1, "SGB_D": 1}, idx)
    rows += r
    r, idx = rows_for("R4", "S2", {"SGB_A": 1, "SGB_B": 1, "SGB_D": 2}, idx)
    rows += r
    r, idx = rows_for("R4", "S3", {"SGB_A": 2, "SGB_C": 1}, idx)
    rows += r

    return pd.DataFrame(rows)


def _write_very_complex_blast_file(tmp_path: Path, mapping_df: pd.DataFrame) -> Path:
    """Write BLAST rows for a set of inter-SGB MAG pairs spanning all levels.

    Only unique directions are encoded; canonicalization de-duplicates reversed pairs.
    """

    # Build a quick index from mapping to find MAG IDs per (replicate, sample, SGB)
    def mags(repl, samp, sgb):
        return mapping_df[
            (mapping_df["replicate"] == repl)
            & (mapping_df["sample_id"] == f"{repl}_{samp}")
            & (mapping_df["SGB"] == sgb)
        ]["MAG_ID"].tolist()

    pairs = []
    # Within mouse (R1_S1): choose some A-B, A-C, B-C, A-D
    a_r1s1 = mags("R1", "S1", "SGB_A")
    b_r1s1 = mags("R1", "S1", "SGB_B")
    c_r1s1 = mags("R1", "S1", "SGB_C")
    d_r1s1 = mags("R1", "S1", "SGB_D")
    pairs += [
        (a_r1s1[0], b_r1s1[0]),
        (a_r1s1[0], c_r1s1[0]),
        (b_r1s1[1], c_r1s1[0]),
        (a_r1s1[1], d_r1s1[0]),
    ]

    # Within replicate R1 (cross samples): A (R1_S1) vs B (R1_S2)
    a_r1s1_any = a_r1s1[1]
    b_r1s2 = mags("R1", "S2", "SGB_B")
    pairs += [(a_r1s1_any, b_r1s2[0])]

    # Within replicate R2 (cross samples): A (R2_S2) vs B (R2_S1)
    a_r2s2 = mags("R2", "S2", "SGB_A")[0]
    b_r2s1 = mags("R2", "S1", "SGB_B")[0]
    pairs += [(a_r2s2, b_r2s1)]

    # Within mouse (R2_S3): A vs B
    a_r2s3 = mags("R2", "S3", "SGB_A")[0]
    b_r2s3 = mags("R2", "S3", "SGB_B")[0]
    pairs += [(a_r2s3, b_r2s3)]

    # Between replicates
    # R1 vs R2: A(R1_S1) vs B(R2_S1)
    b_r2s1_all = mags("R2", "S1", "SGB_B")
    pairs += [(a_r1s1[0], b_r2s1_all[0])]
    # R1 vs R3: A(R1_S2) vs C(R3_S1)
    a_r1s2 = mags("R1", "S2", "SGB_A")[0]
    c_r3s1 = mags("R3", "S1", "SGB_C")[0]
    pairs += [(a_r1s2, c_r3s1)]
    # R2 vs R4: D(R2_S2) vs B(R4_S1)
    d_r2s2 = mags("R2", "S2", "SGB_D")[0]
    b_r4s1 = mags("R4", "S1", "SGB_B")[0]
    pairs += [(d_r2s2, b_r4s1)]
    # R3 vs R4: A(R3_S2) vs D(R4_S2)
    a_r3s2 = mags("R3", "S2", "SGB_A")[0]
    d_r4s2 = mags("R4", "S2", "SGB_D")[0]
    pairs += [(a_r3s2, d_r4s2)]

    def make_row(a, b):
        return [
            f"{a}.fa_contig_1",
            f"{b}.fa_contig_2",
            100.0,
            600,
            0,
            0,
            1,
            600,
            10,
            610,
            0.0,
            500,
        ]

    df = pd.DataFrame([make_row(a, b) for a, b in pairs])
    blast_path = tmp_path / "blast_very_complex.tsv"
    df.to_csv(blast_path, sep="\t", header=False, index=False)
    return blast_path, pairs


def test_process_blast_file_and_summaries_very_complex(tmp_path):
    """End-to-end check on a dataset with 4 replicates, 3 samples each, and rich SGB composition.

    Validates selected numerators/denominators across levels for specific SGB pairs.
    """
    mapping_df = _build_very_complex_mapping_df()
    blast_path, pairs = _write_very_complex_blast_file(tmp_path, mapping_df)

    # Build contig map from all MAGs in mapping
    contig_map_df = _build_contig_map_df(mapping_df["MAG_ID"].unique())

    results, inter_df = bha.process_blast_file(
        blast_path=blast_path,
        map_df=mapping_df,
        group="g1",
        timepoint="t1",
        pident=100.0,
        evalue=0.0,
        length=500,
        contig_map_df=contig_map_df,
    )

    # Sanity checks
    assert not inter_df.empty
    assert inter_df["canonical_pair_key"].nunique() == len(pairs)
    assert {"within_mouse", "within_replicate", "between_replicates"}.issubset(
        set(inter_df["hgt_category"].unique())
    )

    # Global level detailed summary
    global_det = results["global"]["detailed_summary"]
    assert global_det.empty

    # Global level aggregated summary for SGB pairs
    global_agg = results["global"]["aggregated_summary"]
    assert not global_agg.empty
    # For A-B as already tested
    g_ab = global_agg[
        (global_agg["SGB_MAG1"] == "SGB_A") & (global_agg["SGB_MAG2"] == "SGB_B")
    ].iloc[0]
    assert int(g_ab["sum_total_possible_mag_pairs"]) == 16 * 13  # A=16, B=13
    # 1 within R1_S1, 1 within R2_S3, 1 within-rep R1, 1 within-rep R2, 1 between R1-R2
    assert int(g_ab["sum_interacting_mag_pairs"]) == 5

    # Within-mouse detailed for R1_S1 and pair (SGB_A, SGB_B)
    within_mouse_det = results["within_mouse"]["detailed_summary"]
    r1s1_ab = within_mouse_det[
        (within_mouse_det["context_id"] == "R1_S1")
        & (within_mouse_det["SGB_MAG1"] == "SGB_A")
        & (within_mouse_det["SGB_MAG2"] == "SGB_B")
    ].iloc[0]
    # R1_S1 has A=2, B=2 -> denom 4; we added one A-B hit
    assert int(r1s1_ab["interacting_mag_pairs"]) == 1
    assert int(r1s1_ab["total_possible_mag_pairs"]) == 4
    assert np.isclose(r1s1_ab["percentage_interacting"], 25.0)

    # Within-mouse aggregated for (SGB_A, SGB_B): sum over all mice
    within_mouse_agg = results["within_mouse"]["aggregated_summary"]
    wm_ab = within_mouse_agg[
        (within_mouse_agg["SGB_MAG1"] == "SGB_A")
        & (within_mouse_agg["SGB_MAG2"] == "SGB_B")
    ].iloc[0]
    assert int(wm_ab["sum_interacting_mag_pairs"]) == 2  # R1_S1 and R2_S3
    assert int(wm_ab["sum_total_number_of_HGTs"]) == 2
    assert (
        int(wm_ab["sum_total_possible_mag_pairs"]) == 16
    )  # Sum of denom across samples with both A and B
    samples = within_mouse_det[
        (within_mouse_det["SGB_MAG1"] == "SGB_A")
        & (within_mouse_det["SGB_MAG2"] == "SGB_B")
    ]
    # 12 within mice samples have both A and B
    assert len(samples) == 12
    assert samples["total_possible_mag_pairs"].sum() == 16
    assert np.isclose(
        wm_ab["mean_interacting_mag_pairs"], 2 / 12
    )  # 2 interacting / 12 samples total
    assert np.isclose(wm_ab["mean_total_number_of_HGTs"], 2 / 12)  # Same as above
    assert np.isclose(
        wm_ab["mean_total_possible_mag_pairs"], 16 / 12
    )  # 16 / 12 samples

    # Within-replicate detailed for R1 and pair (SGB_A, SGB_B)
    within_rep_det = results["within_replicate"]["detailed_summary"]
    r1_ab = within_rep_det[
        (within_rep_det["context_id"] == "R1")
        & (within_rep_det["SGB_MAG1"] == "SGB_A")
        & (within_rep_det["SGB_MAG2"] == "SGB_B")
    ].iloc[0]
    # R1: A totals=5 (2+1+2), B totals=3 (2+1); possible = 5*3 - diagonal sum
    # Diagonal: R1_S1: 2*2=4, R1_S2:1*1=1, R1_S3:2*0=0 -> diag sum=5
    # So denom = 15 - 5 = 10; we have one within rep hit for A-B in R1
    assert int(r1_ab["interacting_mag_pairs"]) == 1
    assert int(r1_ab["total_possible_mag_pairs"]) == 10
    assert int(r1_ab["total_number_of_HGTs"]) == 1
    assert np.isclose(r1_ab["percentage_interacting"], 10.0)

    # Within-replicate aggregated for (SGB_A, SGB_B): contributions from R1 and R2
    within_rep_agg = results["within_replicate"]["aggregated_summary"]
    wr_ab = within_rep_agg[
        (within_rep_agg["SGB_MAG1"] == "SGB_A")
        & (within_rep_agg["SGB_MAG2"] == "SGB_B")
    ].iloc[0]
    # Denominators computed as totals product minus within-mouse diagonals
    # R1: totals A=5, B=3, diag=5 => denom=10; 1 hit
    # R2: totals A=4, B=4, diag=5 => denom=11; 1 hit
    # R3: totals A=4, B=3, diag=5 => denom=7; 0 hits
    # R4: totals A=3, B=3, diag=(0+1+0)=1 => denom=8; 0 hits
    assert int(wr_ab["sum_interacting_mag_pairs"]) == 2
    assert int(wr_ab["sum_total_possible_mag_pairs"]) == (10 + 11 + 7 + 8)
    assert int(wr_ab["sum_total_number_of_HGTs"]) == 2  # same as interacting for A-B
    assert np.isclose(
        wr_ab["mean_interacting_mag_pairs"], 2 / 4
    )  # 0.5 across 4 replicates
    assert np.isclose(wr_ab["mean_total_number_of_HGTs"], 2 / 4)  # 0.5
    assert np.isclose(wr_ab["mean_total_possible_mag_pairs"], 36 / 4)  # 9.0
    samples = within_rep_det[
        (within_rep_det["SGB_MAG1"] == "SGB_A")
        & (within_rep_det["SGB_MAG2"] == "SGB_B")
    ]
    assert len(samples) == 4  # 4 replicates have both A and B
    assert samples["total_possible_mag_pairs"].sum() == 36

    # Between-replicates detailed for (R1,R2) and (SGB_A, SGB_B)
    between_det = results["between_replicates"]["detailed_summary"]
    r12_ab = between_det[
        (between_det["context_id"].astype(str) == str(("R1", "R2")))
        & (between_det["SGB_MAG1"] == "SGB_A")
        & (between_det["SGB_MAG2"] == "SGB_B")
    ].iloc[0]
    # R1: A=5, B=3; R2: A=4, B=4 -> denom = 5*4 + 4*3 = 32; we added 1 A-B hit
    assert int(r12_ab["interacting_mag_pairs"]) == 1
    assert int(r12_ab["total_possible_mag_pairs"]) == 32
    assert np.isclose(r12_ab["percentage_interacting"], (1 / 32) * 100)
    samples = between_det[
        (between_det["SGB_MAG1"] == "SGB_A") & (between_det["SGB_MAG2"] == "SGB_B")
    ]
    assert len(samples) == 6  # 6 between-replicate pairs have both
    assert samples["total_possible_mag_pairs"].sum() == 156

    # Between-replicates aggregated for (SGB_A, SGB_B)
    between_rep_agg = results["between_replicates"]["aggregated_summary"]
    br_ab = between_rep_agg[
        (between_rep_agg["SGB_MAG1"] == "SGB_A")
        & (between_rep_agg["SGB_MAG2"] == "SGB_B")
    ].iloc[0]
    assert int(br_ab["sum_interacting_mag_pairs"]) == 1  # Only R1-R2 has hits
    assert int(br_ab["sum_total_number_of_HGTs"]) == 1
    assert (
        int(br_ab["sum_total_possible_mag_pairs"])
        == 156  # Sum across all replicate pair contexts
    )
    assert np.isclose(br_ab["mean_interacting_mag_pairs"], 1 / 6, atol=1e-6)  # 1/6
    assert np.isclose(br_ab["mean_total_number_of_HGTs"], 1 / 6, atol=1e-6)  # 1/6
    assert np.isclose(
        br_ab["mean_total_possible_mag_pairs"], 156 / 6, atol=1e-6
    )  # 156/6


def test_perform_count_based_tests():
    # Build a minimal comp_df with the required columns for two conditions
    suffix1 = "_g1_t1"
    suffix2 = "_g2_t1"
    data = {
        "SGB_MAG1": ["SGB_X"],
        "SGB_MAG2": ["SGB_Y"],
        f"sum_interacting_mag_pairs{suffix1}": [3],
        f"sum_total_possible_mag_pairs{suffix1}": [10],
        f"sum_interacting_mag_pairs{suffix2}": [1],
        f"sum_total_possible_mag_pairs{suffix2}": [10],
        f"percentage_interacting{suffix1}": [30.0],
        f"percentage_interacting{suffix2}": [10.0],
    }
    comp_df = pd.DataFrame(data)

    # Focus on suffix1 vs baseline suffix2
    out = bha.perform_count_based_tests(
        comp_df.copy(), focus_suffix=suffix1, baseline_suffix=suffix2
    )
    assert {
        "p_high_binomial",
        "p_low_binomial",
        "p_high_poisson",
        "p_low_poisson",
        "effect_direction",
    }.issubset(out.columns)
    assert out.loc[0, "effect_direction"] in {"enriched", "depleted", "no_change"}


def test_perform_distributional_tests():
    # Build minimal detailed summaries for a single SGB pair across contexts
    d1 = pd.DataFrame(
        {
            "context_id": ["c1", "c2", "c3"],
            "SGB_MAG1": ["SGB_X"] * 3,
            "SGB_MAG2": ["SGB_Y"] * 3,
            "percentage_interacting": [10.0, 20.0, 30.0],
        }
    )
    d2 = pd.DataFrame(
        {
            "context_id": ["c1", "c2", "c3"],
            "SGB_MAG1": ["SGB_X"] * 3,
            "SGB_MAG2": ["SGB_Y"] * 3,
            "percentage_interacting": [10.0, 20.0, 35.0],
        }
    )

    res = bha.perform_distributional_tests(d1, d2, level="within_mouse", min_samples=2)
    assert set(
        [
            "SGB_MAG1",
            "SGB_MAG2",
            "mann_whitney_u_p_value",
            "independent_t_test_p_value",
            "wilcoxon_signed_rank_p_value",
            "paired_t_test_p_value",
        ]
    ).issubset(res.columns)
    assert not res.empty
