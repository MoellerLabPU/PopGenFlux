#!/usr/bin/env python3
"""
HGT analysis test suite for `blast_hgt_analysis.py`.

What this tests:
- Parsing utilities: `extract_mag_id`, `_canonical_pair`
- End-to-end processing on simple and complex synthetic datasets
  covering `within_mouse`, `within_replicate`, `between_replicates`, and
  `global` summaries, validating both sums and means
- Count-based statistical tests (binomial/poisson) and distributional tests

How to run (from the repository root):
- Preferred: `python3 -m pytest tests/test_blast_hgt_analysis.py`
- All tests: `python3 -m pytest`

If pytest is missing:
- `python3 -m pip install --user pytest`

Dependencies: Python 3, pytest, numpy, pandas, scipy, matplotlib
"""
import os
from pathlib import Path
import numpy as np
import pandas as pd

import popgenflux.blast_hgt_analysis as bha
import pytest


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

    # Global aggregated expectations
    global_agg = results["global"]["aggregated_summary"]
    assert not global_agg.empty
    row = global_agg[
        (global_agg["SGB_MAG1"] == "SGB_X") & (global_agg["SGB_MAG2"] == "SGB_Y")
    ].iloc[0]
    assert int(row["sum_interacting_mag_pairs"]) == 3  # A-B, A-D, A-F
    assert int(row["sum_total_possible_mag_pairs"]) == 9  # 3 X by 3 Y
    assert np.isclose(row["percentage_interacting"], (3 / 9) * 100)

    # Within mouse (s1)
    within_mouse_det = results["within_mouse"]["detailed_summary"]
    wm_row = within_mouse_det[(within_mouse_det["context_id"] == "s1")].iloc[0]
    assert int(wm_row["interacting_mag_pairs"]) == 1
    assert int(wm_row["total_number_of_HGTs"]) == 1
    assert int(wm_row["total_possible_mag_pairs"]) == 1
    assert np.isclose(wm_row["percentage_interacting"], 100.0)

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
    assert (
        int(wr_row["sum_total_possible_mag_pairs"]) == 2
    )  # s1-s2 pair contributes 2 ordered combinations. MAG_A-MAG_D and MAG_B-MAG_C
    assert np.isclose(wr_row["mean_interacting_mag_pairs"], 1.0)
    assert np.isclose(wr_row["percentage_interacting"], (1 / 2) * 100)

    # Between replicates aggregated
    between_rep_agg = results["between_replicates"]["aggregated_summary"]
    br_row = between_rep_agg[
        (between_rep_agg["SGB_MAG1"] == "SGB_X")
        & (between_rep_agg["SGB_MAG2"] == "SGB_Y")
    ].iloc[0]
    assert int(br_row["sum_interacting_mag_pairs"]) == 1
    assert int(br_row["sum_total_possible_mag_pairs"]) == 4  # (2*1)+(1*2)
    assert np.isclose(br_row["mean_interacting_mag_pairs"], 1.0)
    assert np.isclose(br_row["percentage_interacting"], (1 / 4) * 100)


def _build_complex_mapping_df() -> pd.DataFrame:
    """Complex dataset per the user's scenario: 4 mice across 3 replicates.

    Replicate R1
      - Mouse M1: SGB_X (A1, A2), SGB_Y (B1)
    Replicate R2
      - Mouse M2: SGB_X (A3), SGB_Y (B2)
      - Mouse M3: SGB_X (A4), SGB_Y (B3, B4)
    Replicate R3
      - Mouse M4: SGB_X (A5), SGB_Y (B5)

    Totals: X has 5 MAGs, Y has 5 MAGs.
    """
    # Complex dataset per user's scenario: 4 mice across 3 replicates
    rows = [
        # R1 - M1: X(A1,A2), Y(B1)
        {
            "MAG_ID": "A1",
            "SGB": "SGB_X",
            "sample_id": "M1",
            "replicate": "R1",
            "subjectID": "sub1",
            "time": "t1",
            "group": "g1",
        },
        {
            "MAG_ID": "A2",
            "SGB": "SGB_X",
            "sample_id": "M1",
            "replicate": "R1",
            "subjectID": "sub1",
            "time": "t1",
            "group": "g1",
        },
        {
            "MAG_ID": "B1",
            "SGB": "SGB_Y",
            "sample_id": "M1",
            "replicate": "R1",
            "subjectID": "sub1",
            "time": "t1",
            "group": "g1",
        },
        # R2 - M2: X(A3), Y(B2)
        {
            "MAG_ID": "A3",
            "SGB": "SGB_X",
            "sample_id": "M2",
            "replicate": "R2",
            "subjectID": "sub2",
            "time": "t1",
            "group": "g1",
        },
        {
            "MAG_ID": "B2",
            "SGB": "SGB_Y",
            "sample_id": "M2",
            "replicate": "R2",
            "subjectID": "sub2",
            "time": "t1",
            "group": "g1",
        },
        # R2 - M3: X(A4), Y(B3,B4)
        {
            "MAG_ID": "A4",
            "SGB": "SGB_X",
            "sample_id": "M3",
            "replicate": "R2",
            "subjectID": "sub3",
            "time": "t1",
            "group": "g1",
        },
        {
            "MAG_ID": "B3",
            "SGB": "SGB_Y",
            "sample_id": "M3",
            "replicate": "R2",
            "subjectID": "sub3",
            "time": "t1",
            "group": "g1",
        },
        {
            "MAG_ID": "B4",
            "SGB": "SGB_Y",
            "sample_id": "M3",
            "replicate": "R2",
            "subjectID": "sub3",
            "time": "t1",
            "group": "g1",
        },
        # R3 - M4: X(A5), Y(B5)
        {
            "MAG_ID": "A5",
            "SGB": "SGB_X",
            "sample_id": "M4",
            "replicate": "R3",
            "subjectID": "sub4",
            "time": "t1",
            "group": "g1",
        },
        {
            "MAG_ID": "B5",
            "SGB": "SGB_Y",
            "sample_id": "M4",
            "replicate": "R3",
            "subjectID": "sub4",
            "time": "t1",
            "group": "g1",
        },
    ]
    return pd.DataFrame(rows)


def _write_complex_blast_file(tmp_path: Path) -> Path:
    """Write BLAST rows for the 7 unique interacting MAG pairs.

    We intentionally encode only the "unique" direction (A,B) and rely on the
    canonical-pair logic to ensure counts reflect unique MAG-MAG interactions.
    All rows meet the filtering thresholds.
    """
    # Observed unique interacting MAG pairs (as qseqid, sseqid)
    pairs = [
        ("A1", "B1"),  # within M1
        ("A2", "B1"),  # within M1
        ("A4", "B3"),  # within M3
        ("A3", "B4"),  # between M2 and M3 within R2
        ("A1", "B3"),  # between R1 and R2
        ("A5", "B1"),  # between R3 and R1
        ("A5", "B3"),  # between R3 and R2
    ]

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
    blast_path = tmp_path / "blast_complex.tsv"
    df.to_csv(blast_path, sep="\t", header=False, index=False)
    return blast_path


def test_process_blast_file_and_summaries_complex(tmp_path):
    """End-to-end check on the complex 4-mice/3-replicate dataset.

    Asserts the exact numerators and denominators described in the spec for:
    - Within-mouse (M1 and M3)
    - Within-replicate (R2 only)
    - Between-replicates ((R1,R2), (R1,R3), (R2,R3))
    - Global (X vs Y across all MAGs)
    """
    map_df = _build_complex_mapping_df()
    blast_path = _write_complex_blast_file(tmp_path)

    contig_map_df = _build_contig_map_df(
        ["A1", "A2", "A3", "A4", "A5", "B1", "B2", "B3", "B4", "B5"]
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

    # Sanity checks
    assert not inter_df.empty
    assert inter_df["canonical_pair_key"].nunique() == 7
    assert set(inter_df["hgt_category"].unique()) == {
        "within_mouse",
        "within_replicate",
        "between_replicates",
    }

    # Level 1: Within-Mouse detailed
    # Denominator per mouse is (#X in mouse) × (#Y in mouse)
    # Numerator is the number of unique interacting MAG pairs for that mouse.
    within_mouse_det = results["within_mouse"]["detailed_summary"]
    # M1: X=2, Y=1 -> denom=2, numerator=2
    m1 = within_mouse_det[within_mouse_det["context_id"] == "M1"].iloc[0]
    assert int(m1["interacting_mag_pairs"]) == 2
    assert int(m1["total_possible_mag_pairs"]) == 2
    assert np.isclose(m1["percentage_interacting"], 100.0)
    # M3: X=1, Y=2 -> denom=2, numerator=1
    m3 = within_mouse_det[within_mouse_det["context_id"] == "M3"].iloc[0]
    assert int(m3["interacting_mag_pairs"]) == 1
    assert int(m3["total_possible_mag_pairs"]) == 2
    assert np.isclose(m3["percentage_interacting"], 50.0)

    # Within-Mouse aggregated
    within_mouse_agg = results["within_mouse"]["aggregated_summary"]
    agg_row = within_mouse_agg[
        (within_mouse_agg["SGB_MAG1"] == "SGB_X")
        & (within_mouse_agg["SGB_MAG2"] == "SGB_Y")
    ].iloc[0]
    assert int(agg_row["sum_interacting_mag_pairs"]) == 3
    assert int(agg_row["sum_total_possible_mag_pairs"]) == 4
    assert np.isclose(agg_row["percentage_interacting"], (3 / 4) * 100)
    # Mean checks across M1 (2/2) and M3 (1/2)
    assert np.isclose(agg_row["mean_interacting_mag_pairs"], 1.5)
    assert np.isclose(agg_row["mean_total_number_of_HGTs"], 1.5)
    assert np.isclose(agg_row["mean_total_possible_mag_pairs"], 2.0)

    # Level 2: Within-Replicate aggregated (only R2 contributes)
    # R2 denominator: mice M2 (X=1,Y=1) and M3 (X=1,Y=2) -> (1×2)+(1×1)=3
    within_rep_agg = results["within_replicate"]["aggregated_summary"]
    wr_row = within_rep_agg[
        (within_rep_agg["SGB_MAG1"] == "SGB_X")
        & (within_rep_agg["SGB_MAG2"] == "SGB_Y")
    ].iloc[0]
    assert int(wr_row["sum_interacting_mag_pairs"]) == 1
    assert int(wr_row["sum_total_possible_mag_pairs"]) == 3  # (1*2)+(1*1)
    assert np.isclose(wr_row["percentage_interacting"], (1 / 3) * 100)
    # Only R2 contributes, so means equal the single-row values
    assert np.isclose(wr_row["mean_interacting_mag_pairs"], 1.0)
    assert np.isclose(wr_row["mean_total_number_of_HGTs"], 1.0)
    assert np.isclose(wr_row["mean_total_possible_mag_pairs"], 3.0)

    # Level 3: Between-Replicates detailed and aggregated
    # Denominator for (R1,R2): X_R1=2, Y_R1=1, X_R2=2, Y_R2=3 => (2×3)+(2×1)=8
    # Denominator for (R1,R3): X_R3=1, Y_R3=1 => (2×1)+(1×1)=3
    # Denominator for (R2,R3): (2×1)+(1×3)=5
    between_det = results["between_replicates"]["detailed_summary"]
    # (R1,R2)
    r12 = between_det[between_det["context_id"].astype(str) == str(("R1", "R2"))].iloc[
        0
    ]
    assert int(r12["interacting_mag_pairs"]) == 1
    assert int(r12["total_possible_mag_pairs"]) == 8  # (2*3)+(2*1)
    # (R1,R3)
    r13 = between_det[between_det["context_id"].astype(str) == str(("R1", "R3"))].iloc[
        0
    ]
    assert int(r13["interacting_mag_pairs"]) == 1
    assert int(r13["total_possible_mag_pairs"]) == 3  # (2*1)+(1*1)
    # (R2,R3)
    r23 = between_det[between_det["context_id"].astype(str) == str(("R2", "R3"))].iloc[
        0
    ]
    assert int(r23["interacting_mag_pairs"]) == 1
    assert int(r23["total_possible_mag_pairs"]) == 5  # (2*1)+(1*3)

    between_agg = results["between_replicates"]["aggregated_summary"]
    ba_row = between_agg[
        (between_agg["SGB_MAG1"] == "SGB_X") & (between_agg["SGB_MAG2"] == "SGB_Y")
    ].iloc[0]
    assert int(ba_row["sum_interacting_mag_pairs"]) == 3
    assert int(ba_row["sum_total_possible_mag_pairs"]) == 16  # 8+3+5
    assert np.isclose(ba_row["percentage_interacting"], (3 / 16) * 100)
    # Means across three replicate pairs: numerators all 1; denominators 8,3,5
    assert np.isclose(ba_row["mean_interacting_mag_pairs"], 1.0)
    assert np.isclose(ba_row["mean_total_number_of_HGTs"], 1.0)
    assert np.isclose(ba_row["mean_total_possible_mag_pairs"], 16.0 / 3.0)

    # Level 4: Global aggregated
    # 7 unique interacting MAG pairs across the dataset; denominator is 5×5=25.
    global_agg = results["global"]["aggregated_summary"]
    g_row = global_agg[
        (global_agg["SGB_MAG1"] == "SGB_X") & (global_agg["SGB_MAG2"] == "SGB_Y")
    ].iloc[0]
    assert int(g_row["sum_interacting_mag_pairs"]) == 7
    assert int(g_row["sum_total_possible_mag_pairs"]) == 25  # 5 X * 5 Y
    assert np.isclose(g_row["percentage_interacting"], (7 / 25) * 100)


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
