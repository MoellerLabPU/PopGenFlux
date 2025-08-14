#!/usr/bin/env python3
"""
HGT comparison and significance analysis from two BLAST result files.

What this script does
- Ingests two BLASTN tabular result files (inter-contig alignments) and a MAG→SGB mapping.
- Filters hits by identity, e-value, and alignment length; removes self-hits and duplicate A↔B pairs.
- Maps MAGs to context (SGB, sample_id, replicate, group, time) and classifies hits into levels:
  global, within_mouse, within_replicate, between_replicates.
- Summarizes per SGB-pair: interacting MAG-pair counts, possible MAG-pair denominators, and percentages.
- Compares conditions using count-based tests (Binomial, Poisson) and optional distributional tests (MWU, t-tests).
- Saves detailed/aggregated tables, comparison statistics, and optional plots to an output directory.

Input requirements
- BLAST files must be TSV with columns (outfmt 6) exactly:
  qseqid sseqid pident length mismatch gapopen qstart qend sstart send evalue bitscore
  Example BLASTN command to generate compatible output:
  blastn -query QUERY.fa -subject SUBJECT.fa \
         -outfmt "6 qseqid sseqid pident length mismatch gapopen qstart qend sstart send evalue bitscore" \
         -out /abs/path/blast_results.tsv
- Mapping file must be a TSV containing columns: MAG_ID, SGB, sample_id, replicate, subjectID, time, group.

How to run (example)
python3 /home/suppal/DietDrivenMicrobiota/blast_hgt_analysis.py \
  --blast1 /abs/path/blast_groupA_T0.tsv --group1 GroupA --timepoint1 T0 \
  --blast2 /abs/path/blast_groupB_T0.tsv --group2 GroupB --timepoint2 T0 \
  --focus-group GroupA \
  --mapping_file /abs/path/mag_to_sgb_mapping.tsv \
  --pident 100 --evalue 0 --length 500 \
  --plot --stat_plot_type poisson --alpha 0.05 \
  --output-dir /abs/path/hgt_analysis_results --prefix hgt_cmp --min-samples 2

Outputs
- <prefix>_classified_hits_<group>_<time>.tsv: Classified inter-SGB hits per input file.
- <prefix>_<level>_detailed_summary_<group>_<time>.tsv: Context-stratified summaries.
- <prefix>_<level>_aggregated_summary_<group>_<time>.tsv: Aggregated summaries.
- <prefix>_<level>_statistical_comparison.tsv: Merged comparisons with p-values.
- Plots (*.png) if --plot is provided.

Dependencies
- Python 3.9+; install packages: pandas, numpy, scipy, matplotlib.
  Example: pip install pandas numpy scipy matplotlib
"""
import argparse
import logging
import os
import sys
from itertools import combinations
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import binom, mannwhitneyu, poisson, ttest_ind, ttest_rel, wilcoxon

# --- Configuration ---
# Define the column headers for the BLASTN output file. This provides a clear,
# readable structure for the data loaded from the BLAST results.
BLAST_COLS = [
    "qseqid",
    "sseqid",
    "pident",
    "length",
    "mismatch",
    "gapopen",
    "qstart",
    "qend",
    "sstart",
    "send",
    "evalue",
    "bitscore",
]

# Set up a global logger for the script. This allows logging from any function.
logger = logging.getLogger(__name__)


def _map_seqids_to_mag_ids(
    seqids: pd.Series,
    contig_map_df: pd.DataFrame,
) -> pd.Series:
    """Map BLAST sequence IDs (gene IDs) to MAG IDs via contig→MAG map.

    Expected gene ID shape: contigID_geneID (contig and gene separated by an underscore).
    Mapping uses the prefix before the last underscore as the contig ID.

    The contig map must contain columns 'contig_id' and 'mag_id'.

    Raises a SystemExit if any sequence ID cannot be mapped to a MAG.
    """
    if (
        "contig_id" not in contig_map_df.columns
        or "mag_id" not in contig_map_df.columns
    ):
        raise ValueError(
            "Contig map is missing required columns 'contig_id' and/or 'mag_id'."
        )

    contig_to_mag = contig_map_df[["contig_id", "mag_id"]].dropna()
    # Ensure each contig_id maps to a single MAG
    dup = contig_to_mag["contig_id"].duplicated(keep=False)
    if dup.any():
        duplicated_keys = contig_to_mag.loc[dup, "contig_id"].unique()
        raise SystemExit(
            "Contig-to-MAG mapping contains duplicate contig IDs mapping to multiple MAGs: "
            f"{duplicated_keys[:10]}{'…' if len(duplicated_keys) > 10 else ''}"
        )

    contig_to_mag_dict = dict(
        zip(
            contig_to_mag["contig_id"].astype(str),
            contig_to_mag["mag_id"].astype(str),
        )
    )

    # Determine contig candidates from sequence IDs using prefix before last underscore
    seqids_str = seqids.astype(str)
    contig_candidates = seqids_str.map(lambda s: s.rsplit("_", 1)[0] if "_" in s else s)
    mags = contig_candidates.map(contig_to_mag_dict)

    # Strict behavior: any unmapped IDs cause an error
    unmapped_mask = mags.isna()
    if unmapped_mask.any():
        unmapped_examples = seqids_str[unmapped_mask].unique()[:10]
        total_unmapped = int(unmapped_mask.sum())
        raise SystemExit(
            f"Failed to map {total_unmapped} sequence IDs to MAGs using contig prefixes. "
            f"Examples: {list(unmapped_examples)}. Ensure gene IDs follow 'contigID_geneID' and "
            f"that all contig IDs exist in the contig→MAG map."
        )

    return mags


def _canonical_pair(a: pd.Series, b: pd.Series) -> tuple[pd.Series, pd.Series]:
    """Return lexicographically ordered pairs for two aligned Series.

    Ensures that (A, B) and (B, A) are represented consistently by ordering the
    pair element-wise, which is useful for de-duplicating undirected pairs.

    Args:
        a: First pandas Series.
        b: Second pandas Series aligned with `a`.

    Returns:
        Tuple of two Series `(first, second)` where `first[i] <= second[i]` for
        all valid positions.
    """
    # Create a boolean mask where elements in 'a' are less than or equal to 'b'.
    is_a_min = a <= b
    # Use the pandas-native .where() method which preserves dtypes and indices.
    first = a.where(is_a_min, b)
    second = b.where(is_a_min, a)
    return first, second


def generate_level_summaries(
    inter_sgb_df: pd.DataFrame, context_map_df: pd.DataFrame
) -> dict:
    """Generate detailed and aggregated summaries at multiple analysis levels.

    Computes per-SGB-pair metrics across four levels:
    - global
    - within_mouse (stratified by `sample_id`)
    - within_replicate (stratified by `replicate` excluding within-mouse)
    - between_replicates (pairs of replicates)

    For each level, returns:
    - detailed_summary: per-context rows with counts and denominators
    - aggregated_summary: per-SGB-pair sums/means and denominators

    Expected columns in `inter_sgb_df` include: `SGB1`, `SGB2`, `MAG1`, `MAG2`,
    `sample_id1`, `sample_id2`, `replicate1`, `replicate2`, and a precomputed
    `canonical_pair_key`.

    Args:
        inter_sgb_df: Filtered BLAST hits mapped to context, only inter-SGB rows.
        context_map_df: Mapping rows for the relevant group/time with columns
            `MAG_ID`, `SGB`, `sample_id`, `replicate`.

    Returns:
        Dict keyed by level name where each value is a dict with keys
        `detailed_summary` and `aggregated_summary` holding DataFrames.
    """
    results = {}

    # --- Define Analysis Levels ---
    # Create a dictionary of DataFrames, one for each analysis level.
    # Using .copy() here prevents SettingWithCopyWarning when modifying these slices later.
    analysis_levels = {
        "global": inter_sgb_df,
        "within_mouse": inter_sgb_df[
            inter_sgb_df["hgt_category"] == "within_mouse"
        ].copy(),
        "within_replicate": inter_sgb_df[
            inter_sgb_df["hgt_category"] == "within_replicate"
        ].copy(),
        "between_replicates": inter_sgb_df[
            inter_sgb_df["hgt_category"] == "between_replicates"
        ].copy(),
    }

    # --- Loop Through Each Level ---
    for level, df_subset in analysis_levels.items():
        logger.info(
            f"Calculating summaries for '{level}' level ({len(df_subset):,} hits where SGB1 != SGB2)..."
        )

        if df_subset.empty:
            results[level] = {
                "detailed_summary": pd.DataFrame(),
                "aggregated_summary": pd.DataFrame(),
            }
            continue

        # --- Within-Mouse Level (Stratified) ---
        if level == "within_mouse":
            # Group hits by the specific mouse and SGB pair where they occurred.
            detailed_summary = (
                df_subset.groupby(["sample_id1", "SGB1_canonical", "SGB2_canonical"])
                .agg(
                    interacting_mag_pairs=("canonical_pair_key", "nunique"),
                    total_number_of_HGTs=("canonical_pair_key", "size"),
                )
                .reset_index()
            )

            def get_mouse_denominator(row):
                mouse_context = context_map_df[
                    context_map_df["sample_id"] == row["sample_id1"]
                ]
                sgb1_count = mouse_context[
                    mouse_context["SGB"] == row["SGB1_canonical"]
                ]["MAG_ID"].nunique()
                sgb2_count = mouse_context[
                    mouse_context["SGB"] == row["SGB2_canonical"]
                ]["MAG_ID"].nunique()
                return sgb1_count * sgb2_count

            detailed_summary["total_possible_mag_pairs"] = detailed_summary.apply(
                get_mouse_denominator, axis=1
            )

            detailed_summary.rename(
                columns={
                    "sample_id1": "context_id",
                    "SGB1_canonical": "SGB_MAG1",
                    "SGB2_canonical": "SGB_MAG2",
                },
                inplace=True,
            )

            # Create the aggregated summary by calculating both the sum and mean of the detailed results.
            aggregated_summary = (
                detailed_summary.groupby(["SGB_MAG1", "SGB_MAG2"])
                .agg(
                    sum_interacting_mag_pairs=("interacting_mag_pairs", "sum"),
                    mean_interacting_mag_pairs=("interacting_mag_pairs", "mean"),
                    sum_total_number_of_HGTs=("total_number_of_HGTs", "sum"),
                    mean_total_number_of_HGTs=("total_number_of_HGTs", "mean"),
                    sum_total_possible_mag_pairs=("total_possible_mag_pairs", "sum"),
                    mean_total_possible_mag_pairs=("total_possible_mag_pairs", "mean"),
                )
                .reset_index()
            )

            results[level] = {
                "detailed_summary": detailed_summary,
                "aggregated_summary": aggregated_summary,
            }

        # --- Within-Replicate Level (Stratified) ---
        elif level == "within_replicate":
            stratified_summary = (
                df_subset.groupby(["replicate1", "SGB1_canonical", "SGB2_canonical"])
                .agg(
                    interacting_mag_pairs=("canonical_pair_key", "nunique"),
                    total_number_of_HGTs=("canonical_pair_key", "size"),
                )
                .reset_index()
            )

            def get_replicate_denominator(row):
                rep_context = context_map_df[
                    context_map_df["replicate"] == row["replicate1"]
                ]
                counts_per_mouse = (
                    rep_context.groupby(["sample_id", "SGB"])["MAG_ID"]
                    .nunique()
                    .unstack(fill_value=0)
                )
                total_pairs = 0
                for m1, m2 in combinations(counts_per_mouse.index, 2):
                    m1_sgb1 = counts_per_mouse.loc[m1].get(row["SGB1_canonical"], 0)
                    m1_sgb2 = counts_per_mouse.loc[m1].get(row["SGB2_canonical"], 0)
                    m2_sgb2 = counts_per_mouse.loc[m2].get(row["SGB2_canonical"], 0)
                    m2_sgb1 = counts_per_mouse.loc[m2].get(row["SGB1_canonical"], 0)
                    total_pairs += (m1_sgb1 * m2_sgb2) + (m2_sgb1 * m1_sgb2)
                return total_pairs

            stratified_summary["total_possible_mag_pairs"] = stratified_summary.apply(
                get_replicate_denominator, axis=1
            )

            detailed_summary = stratified_summary.rename(
                columns={
                    "replicate1": "context_id",
                    "SGB1_canonical": "SGB_MAG1",
                    "SGB2_canonical": "SGB_MAG2",
                }
            )

            aggregated_summary = (
                detailed_summary.groupby(["SGB_MAG1", "SGB_MAG2"])
                .agg(
                    sum_interacting_mag_pairs=("interacting_mag_pairs", "sum"),
                    mean_interacting_mag_pairs=("interacting_mag_pairs", "mean"),
                    sum_total_number_of_HGTs=("total_number_of_HGTs", "sum"),
                    mean_total_number_of_HGTs=("total_number_of_HGTs", "mean"),
                    sum_total_possible_mag_pairs=("total_possible_mag_pairs", "sum"),
                    mean_total_possible_mag_pairs=("total_possible_mag_pairs", "mean"),
                )
                .reset_index()
            )

            results[level] = {
                "detailed_summary": detailed_summary,
                "aggregated_summary": aggregated_summary,
            }

        # --- Between-Replicates Level (Stratified) ---
        elif level == "between_replicates":
            df_subset["replicate_pair"] = df_subset.apply(
                lambda row: tuple(sorted((row["replicate1"], row["replicate2"]))),
                axis=1,
            )

            detailed_summary = (
                df_subset.groupby(
                    ["replicate_pair", "SGB1_canonical", "SGB2_canonical"]
                )
                .agg(
                    interacting_mag_pairs=("canonical_pair_key", "nunique"),
                    total_number_of_HGTs=("canonical_pair_key", "size"),
                )
                .reset_index()
            )

            def get_between_replicate_denominator(row):
                rep1_id, rep2_id = row["replicate_pair"]
                sgb1, sgb2 = row["SGB1_canonical"], row["SGB2_canonical"]
                rep1_counts = (
                    context_map_df[context_map_df["replicate"] == rep1_id]
                    .groupby("SGB")["MAG_ID"]
                    .nunique()
                )
                rep2_counts = (
                    context_map_df[context_map_df["replicate"] == rep2_id]
                    .groupby("SGB")["MAG_ID"]
                    .nunique()
                )
                r1_sgb1, r1_sgb2 = rep1_counts.get(sgb1, 0), rep1_counts.get(sgb2, 0)
                r2_sgb1, r2_sgb2 = rep2_counts.get(sgb1, 0), rep2_counts.get(sgb2, 0)
                return (r1_sgb1 * r2_sgb2) + (r2_sgb1 * r1_sgb2)

            detailed_summary["total_possible_mag_pairs"] = detailed_summary.apply(
                get_between_replicate_denominator, axis=1
            )

            detailed_summary.rename(
                columns={
                    "replicate_pair": "context_id",
                    "SGB1_canonical": "SGB_MAG1",
                    "SGB2_canonical": "SGB_MAG2",
                },
                inplace=True,
            )

            aggregated_summary = (
                detailed_summary.groupby(["SGB_MAG1", "SGB_MAG2"])
                .agg(
                    sum_interacting_mag_pairs=("interacting_mag_pairs", "sum"),
                    mean_interacting_mag_pairs=("interacting_mag_pairs", "mean"),
                    sum_total_number_of_HGTs=("total_number_of_HGTs", "sum"),
                    mean_total_number_of_HGTs=("total_number_of_HGTs", "mean"),
                    sum_total_possible_mag_pairs=("total_possible_mag_pairs", "sum"),
                    mean_total_possible_mag_pairs=("total_possible_mag_pairs", "mean"),
                )
                .reset_index()
            )

            results[level] = {
                "detailed_summary": detailed_summary,
                "aggregated_summary": aggregated_summary,
            }

        # --- Global Level (Aggregated Only) ---
        elif level == "global":
            # Aggregate hits for the global level.
            aggregated_summary = (
                df_subset.groupby(["SGB1_canonical", "SGB2_canonical"])
                .agg(
                    sum_interacting_mag_pairs=("canonical_pair_key", "nunique"),
                    sum_total_number_of_HGTs=("canonical_pair_key", "size"),
                )
                .reset_index()
            )

            # Calculate the global denominator.
            sgb_counts = context_map_df.groupby("SGB")["MAG_ID"].nunique()
            aggregated_summary["sum_total_possible_mag_pairs"] = (
                aggregated_summary.apply(
                    lambda row: sgb_counts.get(row["SGB1_canonical"], 0)
                    * sgb_counts.get(row["SGB2_canonical"], 0),
                    axis=1,
                )
            )

            aggregated_summary.rename(
                columns={"SGB1_canonical": "SGB_MAG1", "SGB2_canonical": "SGB_MAG2"},
                inplace=True,
            )
            results[level] = {
                "detailed_summary": pd.DataFrame(),
                "aggregated_summary": aggregated_summary,
            }

        else:
            raise ValueError(f"Unknown analysis level provided: {level}")

        # --- Final Percentage Calculation for all summaries ---
        for summary_type in ["detailed_summary", "aggregated_summary"]:
            summary_df = results[level][summary_type]
            if not summary_df.empty and (
                "total_possible_mag_pairs" in summary_df.columns
                or "sum_total_possible_mag_pairs" in summary_df.columns
            ):
                # Use the sum for the aggregated percentage calculation
                numerator_col = (
                    "sum_interacting_mag_pairs"
                    if "sum_interacting_mag_pairs" in summary_df.columns
                    else "interacting_mag_pairs"
                )
                denominator_col = (
                    "sum_total_possible_mag_pairs"
                    if "sum_total_possible_mag_pairs" in summary_df.columns
                    else "total_possible_mag_pairs"
                )

                summary_df["percentage_interacting"] = np.where(
                    summary_df[denominator_col] > 0,
                    (summary_df[numerator_col] / summary_df[denominator_col]) * 100,
                    0,
                )
    return results


def process_blast_file(
    blast_path: Path,
    map_df: pd.DataFrame,
    group: str,
    timepoint: str,
    pident: float,
    evalue: float,
    length: int,
    contig_map_df: pd.DataFrame,
) -> tuple[dict, pd.DataFrame]:
    """Process a single BLAST file, map to context, classify, and summarize.

    Steps:
    1) Read BLAST TSV (outfmt 6 columns defined in `BLAST_COLS`).
    2) Remove self-hits and duplicate undirected pairs; apply quality filters.
    3) Map MAGs to SGB/sample/replicate for the specified `group` and `timepoint`.
    4) Keep only inter-SGB hits, classify into analysis levels, and summarize.

    Args:
        blast_path: Absolute or relative path to BLAST results TSV.
        map_df: Mapping DataFrame (columns: `MAG_ID`, `SGB`, `sample_id`,
            `replicate`, `subjectID`, `time`, `group`).
        group: Group name for this BLAST file (must match mapping rows).
        timepoint: Timepoint label (must match mapping rows).
        pident: Minimum percent identity threshold.
        evalue: Maximum e-value threshold.
        length: Minimum alignment length threshold.

    Returns:
        Tuple of:
          - results: Dict from `generate_level_summaries` for this file.
          - inter_sgb_df: DataFrame of classified, inter-SGB hits with context.

    Raises:
        SystemExit: If duplicate `MAG_ID` values are present in the mapping for
            the requested context (to prevent erroneous Cartesian merges).
    """
    logger.info(f"Processing {blast_path} [group={group}, time={timepoint}]")

    # --- Load and Filter Data ---
    logger.info("Loading and filtering BLAST hits...")
    blast_df = pd.read_csv(blast_path, sep="\t", header=None, names=BLAST_COLS)

    # Create canonical pairs first to handle self-hits and duplicates robustly.
    blast_df["c_qseqid"], blast_df["c_sseqid"] = _canonical_pair(
        blast_df["qseqid"], blast_df["sseqid"]
    )

    # Filter out self-hits (e.g., A vs A) based on the canonical pair.
    filtered_df = blast_df[blast_df["c_qseqid"] != blast_df["c_sseqid"]].copy()

    # Apply quality filters based on user-defined thresholds.
    mask = (
        (filtered_df["pident"] >= pident)
        & (filtered_df["evalue"] <= evalue)
        & (filtered_df["length"] >= length)
    )
    filtered_df = filtered_df[mask]

    # Drop duplicate hits (e.g., A vs B and B vs A), keeping the first occurrence.
    filtered_df.drop_duplicates(
        subset=["c_qseqid", "c_sseqid"], keep="first", inplace=True
    )

    logger.info(
        f"Retained {len(filtered_df):,} unique, high-quality, non-self alignments."
    )

    # --- Context Mapping ---
    logger.info("Mapping hits to MAG and SGB context...")
    if contig_map_df is None or contig_map_df.empty:
        raise SystemExit(
            "--contig-map is required and must be non-empty. Provide a TSV with columns 'mag_id' and 'contig_id'."
        )
    logger.info("Using contig→MAG mapping file to derive MAG IDs from sequence IDs…")
    filtered_df["MAG1"] = _map_seqids_to_mag_ids(filtered_df["qseqid"], contig_map_df)
    filtered_df["MAG2"] = _map_seqids_to_mag_ids(filtered_df["sseqid"], contig_map_df)

    # Filter the main mapping file to the specific context of this BLAST file.
    context_map_df = map_df[
        (map_df["group"] == group) & (map_df["time"] == timepoint)
    ].copy()

    # --- Validation ---
    # Ensure MAG_IDs are unique within this context to prevent a bad merge, which would
    # create a Cartesian product and massively inflate hit counts.
    if not context_map_df["MAG_ID"].is_unique:
        logger.error(
            f"MAG_IDs are not unique in the mapping file for the context group={group}, time={timepoint}. This can cause incorrect results."
        )
        duplicated_mags = context_map_df[context_map_df["MAG_ID"].duplicated()][
            "MAG_ID"
        ].unique()
        logger.error(f"Duplicated MAGs: {duplicated_mags}")
        sys.exit(1)

    info_cols = ["MAG_ID", "SGB", "sample_id", "replicate"]
    map_subset = context_map_df[info_cols]

    # Merge the filtered BLAST hits with the contextual information. A 'left' merge
    # is used to keep all BLAST hits, even if they don't have a match in the map.
    merged_df = filtered_df.merge(
        map_subset.add_suffix("1"), left_on="MAG1", right_on="MAG_ID1", how="left"
    ).merge(map_subset.add_suffix("2"), left_on="MAG2", right_on="MAG_ID2", how="left")

    # --- Data Cleaning and Validation ---
    # Drop any hits where a MAG could not be mapped to essential context information.
    # This prevents errors in classification and ensures accurate comparisons.
    initial_hit_count = len(merged_df)
    cols_to_check = [
        "SGB1",
        "SGB2",
        "sample_id1",
        "sample_id2",
        "replicate1",
        "replicate2",
    ]
    merged_df.dropna(subset=cols_to_check, inplace=True)
    final_hit_count = len(merged_df)

    # Log a warning if any hits were dropped, as this may indicate an incomplete mapping file.
    if initial_hit_count > final_hit_count:
        dropped_count = initial_hit_count - final_hit_count
        logger.warning(
            f"Dropped {dropped_count:,} hits due to missing mapping information (SGB, sample_id, or replicate). "
            "This may occur if MAGs in the BLAST file are not in the mapping file."
        )

    # --- Hit Classification ---
    logger.info("Classifying hits by analysis level...")
    # This analysis focuses only on interactions between different SGBs.
    inter_sgb_df = merged_df[merged_df["SGB1"] != merged_df["SGB2"]].copy()

    if inter_sgb_df.empty:
        logger.warning("No inter-SGB hits found after context filtering.")
        return {}, pd.DataFrame()

    # Create canonical pairs for SGBs and MAGs for consistent grouping.
    sgb1c, sgb2c = _canonical_pair(inter_sgb_df["SGB1"], inter_sgb_df["SGB2"])
    inter_sgb_df["SGB1_canonical"], inter_sgb_df["SGB2_canonical"] = sgb1c, sgb2c

    mag1c, mag2c = _canonical_pair(inter_sgb_df["MAG1"], inter_sgb_df["MAG2"])
    inter_sgb_df["canonical_pair_key"] = mag1c.str.cat(mag2c, sep="|")

    # Define boolean masks for each analysis level based on sample and replicate IDs.
    within_mouse = inter_sgb_df["sample_id1"] == inter_sgb_df["sample_id2"]
    within_replicate_only = (
        inter_sgb_df["replicate1"] == inter_sgb_df["replicate2"]
    ) & ~within_mouse
    between_replicates = inter_sgb_df["replicate1"] != inter_sgb_df["replicate2"]

    # Use np.select to efficiently assign each hit to a category.
    inter_sgb_df["hgt_category"] = np.select(
        [within_mouse, within_replicate_only, between_replicates],
        ["within_mouse", "within_replicate", "between_replicates"],
        default=None,
    )

    # Check for and log any unclassified rows before dropping them
    unclassified_rows = inter_sgb_df[inter_sgb_df["hgt_category"].isna()]
    if not unclassified_rows.empty:
        logger.warning(
            f"Found {len(unclassified_rows):,} hits that could not be classified into any analysis level. "
            "This may indicate an issue with the replicate or sample_id assignments in the mapping file."
            "Dropping these rows."
        )
    # Drop any rows that couldn't be classified.
    inter_sgb_df.dropna(subset=["hgt_category"], inplace=True)

    # --- Final Summaries ---
    results = generate_level_summaries(inter_sgb_df, context_map_df)

    return results, inter_sgb_df


def perform_count_based_tests(
    comp_df: pd.DataFrame, focus_suffix: str, baseline_suffix: str
) -> pd.DataFrame:
    """Compute one-tailed Binomial and Poisson p-values on aggregated counts.

    Expects `comp_df` to be an inner-merge of aggregated summaries from the two
    conditions. Column names are suffixed (e.g., `sum_interacting_mag_pairs` +
    suffix). Adds columns:
      - `p_high_binomial`, `p_low_binomial`
      - `p_high_poisson`, `p_low_poisson`
      - `effect_direction` in {enriched, depleted, no_change}

    Args:
        comp_df: Merged aggregated DataFrame with required suffixed columns.
        focus_suffix: Suffix for the focus condition (e.g., `_GroupA_T0`).
        baseline_suffix: Suffix for the baseline condition.

    Returns:
        DataFrame with added p-value and effect-direction columns.
    """
    logger.info(
        "Performing count-based statistical comparison (Binomial and Poisson)..."
    )

    focus_hits_col = f"sum_interacting_mag_pairs{focus_suffix}"
    focus_trials_col = f"sum_total_possible_mag_pairs{focus_suffix}"
    baseline_hits_col = f"sum_interacting_mag_pairs{baseline_suffix}"
    baseline_trials_col = f"sum_total_possible_mag_pairs{baseline_suffix}"

    def _calculate_stats(row):
        if row[baseline_trials_col] == 0:
            return pd.Series([np.nan] * 4 + ["No baseline trials"])
        if row[focus_trials_col] == 0:
            return pd.Series([np.nan] * 4 + ["No focus trials"])

        focus_hits = int(row[focus_hits_col])
        focus_trials = int(row[focus_trials_col])
        baseline_hits = int(row[baseline_hits_col])
        baseline_trials = int(row[baseline_trials_col])

        baseline_rate = baseline_hits / baseline_trials
        poisson_mean = focus_trials * baseline_rate

        p_high_binomial = binom.sf(focus_hits - 1, focus_trials, baseline_rate)
        p_low_binomial = binom.cdf(focus_hits, focus_trials, baseline_rate)
        p_high_poisson = poisson.sf(focus_hits - 1, poisson_mean)
        p_low_poisson = poisson.cdf(focus_hits, poisson_mean)

        effect_direction = (
            "enriched"
            if focus_hits > poisson_mean
            else "depleted" if focus_hits < poisson_mean else "no_change"
        )

        return pd.Series(
            [
                p_high_binomial,
                p_low_binomial,
                p_high_poisson,
                p_low_poisson,
                effect_direction,
            ]
        )

    stat_cols = [
        "p_high_binomial",
        "p_low_binomial",
        "p_high_poisson",
        "p_low_poisson",
        "effect_direction",
    ]
    comp_df[stat_cols] = comp_df.apply(_calculate_stats, axis=1)
    return comp_df


def perform_distributional_tests(
    detailed1: pd.DataFrame, detailed2: pd.DataFrame, level: str, min_samples: int
) -> pd.DataFrame:
    """Run distributional tests on context-level interaction percentages.

    Performs both unpaired (Mann-Whitney U, Welch's t-test) and paired
    (Wilcoxon signed-rank, paired t-test) comparisons per SGB pair. Paired
    tests are computed only for contexts present in both groups. Tests are
    performed when each side has at least `min_samples` observations.

    Args:
        detailed1: Detailed summary for condition 1 with `percentage_interacting`.
        detailed2: Detailed summary for condition 2 with `percentage_interacting`.
        level: Analysis level label used for logging.
        min_samples: Minimum observations per group required to run each test.

    Returns:
        DataFrame with one row per SGB pair and columns:
          `mann_whitney_u_p_value`, `independent_t_test_p_value`,
          `wilcoxon_signed_rank_p_value`, `paired_t_test_p_value`.
    """
    logger.info(f"Performing distributional tests for level '{level}'...")

    results = []
    # Build the set of SGB pairs present in either group for unpaired tests.
    sgb_pairs = set()
    if not detailed1.empty:
        sgb_pairs.update(
            map(tuple, detailed1[["SGB_MAG1", "SGB_MAG2"]].dropna().to_numpy())
        )
    if not detailed2.empty:
        sgb_pairs.update(
            map(tuple, detailed2[["SGB_MAG1", "SGB_MAG2"]].dropna().to_numpy())
        )

    for sgb1, sgb2 in sgb_pairs:
        # --- Unpaired data: use all contexts within each group for this SGB pair ---
        g1_unpaired = (
            detailed1[
                (detailed1["SGB_MAG1"] == sgb1) & (detailed1["SGB_MAG2"] == sgb2)
            ]["percentage_interacting"].dropna()
            if not detailed1.empty
            else pd.Series(dtype=float)
        )
        g2_unpaired = (
            detailed2[
                (detailed2["SGB_MAG1"] == sgb1) & (detailed2["SGB_MAG2"] == sgb2)
            ]["percentage_interacting"].dropna()
            if not detailed2.empty
            else pd.Series(dtype=float)
        )

        # --- Paired data: only contexts present in both groups ---
        merged_subset = pd.merge(
            detailed1[
                (detailed1["SGB_MAG1"] == sgb1) & (detailed1["SGB_MAG2"] == sgb2)
            ],
            detailed2[
                (detailed2["SGB_MAG1"] == sgb1) & (detailed2["SGB_MAG2"] == sgb2)
            ],
            on=["context_id", "SGB_MAG1", "SGB_MAG2"],
            how="inner",
            suffixes=("_g1", "_g2"),
        )
        g1_paired = merged_subset["percentage_interacting_g1"]
        g2_paired = merged_subset["percentage_interacting_g2"]

        mwu_p_value, ttest_ind_p_value = np.nan, np.nan
        wilcoxon_p_value, ttest_rel_p_value = np.nan, np.nan

        # --- Unpaired Tests (Mann-Whitney U and Independent t-test) ---
        if len(g1_unpaired) >= min_samples and len(g2_unpaired) >= min_samples:
            mwu_p_value = mannwhitneyu(
                g1_unpaired, g2_unpaired, alternative="two-sided", nan_policy="raise"
            ).pvalue

            # Handle identical-constant distributions yielding NaN p-values in t-test
            if (
                np.mean(g1_unpaired) == np.mean(g2_unpaired)
                and np.var(g1_unpaired) == 0
                and np.var(g2_unpaired) == 0
            ):
                logger.warning(
                    f"No difference found in SGB pair {sgb1} and {sgb2} for level {level}. Setting p-values to 1.0."
                )
                ttest_ind_p_value = 1.0
            else:
                ttest_ind_p_value = ttest_ind(
                    g1_unpaired,
                    g2_unpaired,
                    alternative="two-sided",
                    equal_var=False,
                    nan_policy="raise",
                ).pvalue

        # --- Paired Tests (Wilcoxon and Paired t-test) ---
        if len(g1_paired) >= min_samples and len(g2_paired) >= min_samples:
            diff = g1_paired - g2_paired
            if not all(diff == 0):
                wilcoxon_p_value = wilcoxon(
                    diff, nan_policy="raise", alternative="two-sided"
                ).pvalue
                ttest_rel_p_value = ttest_rel(
                    g1_paired, g2_paired, nan_policy="raise", alternative="two-sided"
                ).pvalue
            else:
                logger.warning(
                    f"No difference found in SGB pair {sgb1} and {sgb2} for level {level}. Setting p-values to 1.0."
                )
                wilcoxon_p_value = 1.0
                ttest_rel_p_value = 1.0

        results.append(
            {
                "SGB_MAG1": sgb1,
                "SGB_MAG2": sgb2,
                "mann_whitney_u_p_value": mwu_p_value,
                "independent_t_test_p_value": ttest_ind_p_value,
                "wilcoxon_signed_rank_p_value": wilcoxon_p_value,
                "paired_t_test_p_value": ttest_rel_p_value,
            }
        )

    return pd.DataFrame(results)


def plot_statistical_results(
    df: pd.DataFrame,
    baseline_suffix: str,
    focus_suffix: str,
    baseline_label: str,
    focus_label: str,
    level: str,
    stat_type: str,
    alpha: float,
    output_dir: Path,
    prefix: str,
):
    """Create a scatter plot colored by statistical significance category.

    Colors SGB-pair points by "Enriched", "Depleted", or "Not Significant"
    based on the chosen count-based test and alpha. The x-axis shows the
    baseline percentage and y-axis the focus percentage.

    Args:
        df: Comparison DataFrame with suffixed percentage and p-value columns.
        baseline_suffix: Suffix for baseline columns in `df`.
        focus_suffix: Suffix for focus columns in `df`.
        baseline_label: Label for baseline condition used in axis titles.
        focus_label: Label for focus condition used in axis titles.
        level: Analysis level name used in the plot title and filename.
        stat_type: Either "binomial" or "poisson" to select p-value columns.
        alpha: Significance threshold.
        output_dir: Directory where the PNG will be saved.
        prefix: Filename prefix for saved plots.
    """
    if df.empty:
        logger.info(
            "DataFrame is empty, skipping statistical plot for level '%s'.", level
        )
        return

    logger.info(
        "Generating statistical significance plot for level '%s' using %s test.",
        level,
        stat_type,
    )

    # Define column names based on user choices.
    p_high_col = f"p_high_{stat_type}"
    p_low_col = f"p_low_{stat_type}"
    baseline_perc_col = f"percentage_interacting{baseline_suffix}"
    focus_perc_col = f"percentage_interacting{focus_suffix}"

    # --- Assign Significance Category ---
    conditions = [
        (df["effect_direction"] == "enriched") & (df[p_high_col] < alpha),
        (df["effect_direction"] == "depleted") & (df[p_low_col] < alpha),
    ]
    choices = ["Enriched", "Depleted"]
    df["significance"] = np.select(conditions, choices, default="Not Significant")

    # --- Plotting ---
    fig, ax = plt.subplots(figsize=(12, 10))
    colors = {"Enriched": "red", "Depleted": "blue", "Not Significant": "grey"}

    # Plot each significance category separately to create the legend.
    for sig_type, color in colors.items():
        subset = df[df["significance"] == sig_type]
        ax.scatter(
            subset[baseline_perc_col],
            subset[focus_perc_col],
            c=color,
            label=sig_type,
            alpha=0.7,
            s=50,
        )

    # Add a y=x line for visual reference (no change).
    lims = [
        np.min([ax.get_xlim(), ax.get_ylim()]),
        np.max([ax.get_xlim(), ax.get_ylim()]),
    ]
    ax.plot(lims, lims, "k--", alpha=0.75, zorder=0)

    # --- Labeling and Formatting ---
    ax.set_xlabel(f"Interaction Percentage ({baseline_label})", fontsize=12)
    ax.set_ylabel(f"Interaction Percentage ({focus_label})", fontsize=12)
    ax.set_title(
        f"Statistical Significance of Interaction Changes ({level.title()})",
        fontsize=16,
    )
    ax.legend(title="Significance")
    ax.grid(True, which="both", linestyle="--", linewidth=0.5)

    plt.tight_layout()

    # --- Save the plot ---
    filename = Path(f"{prefix}_{level}_statistical_plot_{stat_type}.png")
    output_path = output_dir / filename
    plt.savefig(output_path, dpi=300)
    logger.info("Saved statistical plot to %s", output_path)

    if os.getenv("DISPLAY"):
        plt.show()
    plt.close(fig)


def plot_data(
    df: pd.DataFrame,
    y_col_suffix: str,
    y_label: str,
    title: str,
    suffix1: str,
    suffix2: str,
    plot_label1: str,
    plot_label2: str,
    output_dir: Path,
    prefix: str,
):
    """Plot paired values between two conditions for each SGB pair.

    Draws a line for each SGB pair connecting the two condition values and
    colors it by which condition is larger.

    Args:
        df: Comparison DataFrame with the required suffixed columns.
        y_col_suffix: Column root name to plot (suffixes will be appended).
        y_label: Y-axis label.
        title: Plot title, also used to derive the filename.
        suffix1: Suffix for condition 1 columns.
        suffix2: Suffix for condition 2 columns.
        plot_label1: X-axis label for condition 1.
        plot_label2: X-axis label for condition 2.
        output_dir: Directory to save the figure.
        prefix: Filename prefix for saved plots.
    """
    if df.empty:
        logger.info(f"Comparison DataFrame is empty, skipping plot: {title}")
        return

    # Define the column names for the two conditions being plotted.
    y1_col, y2_col = f"{y_col_suffix}{suffix1}", f"{y_col_suffix}{suffix2}"
    x_labels = [plot_label1, plot_label2]

    # Check if the required columns exist in the DataFrame
    if y1_col not in df.columns or y2_col not in df.columns:
        logger.warning(
            f"Columns for plotting ('{y1_col}', '{y2_col}') not found. Skipping plot: {title}"
        )
        return

    # Create the plot figure and axes.
    fig, ax = plt.subplots(figsize=(10, 7))
    # Iterate over each SGB pair to draw a line between the two conditions.
    for index, row in df.iterrows():
        y_values = [row[y1_col], row[y2_col]]
        # Color the line based on which condition has a higher value.
        color = "blue" if y_values[0] > y_values[1] else "red"
        ax.plot(x_labels, y_values, marker="o", linestyle="-", color=color)

    # Set plot labels and title for clarity.
    ax.set_ylabel(y_label, fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.grid(axis="y", linestyle="--", alpha=0.7)
    # Ensure the y-axis starts at 0 for clear interpretation.
    ax.set_ylim(bottom=0)
    plt.tight_layout()

    # --- Save the plot to a file ---
    # Sanitize the title to create a valid filename.
    clean_title = title.replace(" ", "_").replace("(", "").replace(")", "")
    filename = Path(f"{prefix}_{clean_title}.png")
    output_path = output_dir / filename
    plt.savefig(output_path, dpi=300)
    logger.info(f"Saved plot to {output_path}")

    # Only show the plot if in an interactive environment.
    if os.getenv("DISPLAY"):
        plt.show()
    plt.close(fig)


def main():
    """Main entry point for the script."""
    # --- Argument Parsing ---
    # Sets up the command-line interface for the user to provide input files and parameters.
    parser = argparse.ArgumentParser(
        description="Compare HGT events from two BLAST result files across multiple levels.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--blast1",
        required=True,
        type=Path,
        help="Path to the first BLAST results file.",
    )
    parser.add_argument(
        "--group1", required=True, help="Group name for the first file."
    )
    parser.add_argument(
        "--timepoint1", required=True, help="Timepoint for the first file."
    )
    parser.add_argument(
        "--blast2",
        required=True,
        type=Path,
        help="Path to the second BLAST results file.",
    )
    parser.add_argument(
        "--group2", required=True, help="Group name for the second file."
    )
    parser.add_argument(
        "--timepoint2", required=True, help="Timepoint for the second file."
    )
    parser.add_argument(
        "--focus-group",
        required=True,
        help="The group of interest for significance testing (must match --group1 or --group2).",
    )
    parser.add_argument(
        "--mapping_file",
        required=True,
        type=Path,
        help="Path to the MAG-to-SGB mapping TSV file.",
    )
    parser.add_argument(
        "--contig-map",
        required=True,
        type=Path,
        help=(
            "Path to a contig→MAG mapping file (TSV) with columns 'mag_id' and 'contig_id'. "
            "BLAST sequence IDs must follow 'contigID_geneID' (underscore-separated). "
            "Any sequence ID that cannot be mapped using its contig prefix will cause an error."
        ),
    )
    parser.add_argument(
        "--pident", type=float, default=100.0, help="Minimum percent identity."
    )
    parser.add_argument("--evalue", type=float, default=0.0, help="Maximum e-value.")
    parser.add_argument(
        "--length", type=int, default=500, help="Minimum alignment length."
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.05,
        help="Significance level for statistical plotting.",
    )
    parser.add_argument(
        "--stat_plot_type",
        type=str,
        choices=["binomial", "poisson"],
        default="poisson",
        help="Which test to use for coloring the statistical plot.",
    )
    parser.add_argument(
        "--plot",
        action="store_true",
        help="Generate and show plots for each analysis level. Plots will be saved to the output directory.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("hgt_analysis_results"),
        help="Directory to save plots and summary tables.",
    )
    parser.add_argument(
        "--prefix",
        type=str,
        default="hgt_comparison",
        help="Prefix for saved output filenames.",
    )
    parser.add_argument(
        "--min-samples",
        type=int,
        default=2,
        help="Minimum number of samples per group required to run distributional tests.",
    )
    args = parser.parse_args()

    # --- Initial Setup ---
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Validate that the focus group is one of the two provided groups.
    if args.focus_group not in [args.group1, args.group2]:
        logger.error(
            f"Fatal: --focus-group ('{args.focus_group}') must be one of the provided groups ('{args.group1}', '{args.group2}')."
        )
        sys.exit(1)

    # --- Data Loading ---
    # Load the mapping file, which connects MAGs to SGBs and experimental context.
    required_cols = [
        "MAG_ID",
        "SGB",
        "sample_id",
        "replicate",
        "subjectID",
        "time",
        "group",
    ]
    map_df = pd.read_csv(args.mapping_file, usecols=required_cols, sep="\t")

    # Optional contig→MAG mapping
    # Load contig→MAG mapping (required)
    contig_map_df = pd.read_csv(args.contig_map, sep="\t")

    # --- Core Processing ---
    # Process each of the two BLAST files to get summary statistics.
    results1, inter_sgb_df1 = process_blast_file(
        args.blast1,
        map_df,
        args.group1,
        args.timepoint1,
        args.pident,
        args.evalue,
        args.length,
        contig_map_df,
    )
    results2, inter_sgb_df2 = process_blast_file(
        args.blast2,
        map_df,
        args.group2,
        args.timepoint2,
        args.pident,
        args.evalue,
        args.length,
        contig_map_df,
    )

    # --- Comparison Setup ---
    # Create unique suffixes and labels for tables and plots.
    suffix1, suffix2 = (
        f"_{args.group1}_{args.timepoint1}",
        f"_{args.group2}_{args.timepoint2}",
    )
    plot_label1, plot_label2 = (
        f"{args.group1}-{args.timepoint1}",
        f"{args.group2}-{args.timepoint2}",
    )

    if args.focus_group == args.group1:
        focus_suffix, baseline_suffix = suffix1, suffix2
        focus_label, baseline_label = plot_label1, plot_label2
    else:
        focus_suffix, baseline_suffix = suffix2, suffix1
        focus_label, baseline_label = plot_label2, plot_label1

    logger.info(
        f"Focus condition: {focus_label}. Baseline condition: {baseline_label}."
    )

    # Save the detailed classified hits for user inspection.
    if not inter_sgb_df1.empty:
        inter_sgb_df1.to_csv(
            args.output_dir / f"{args.prefix}_classified_hits{suffix1}.tsv",
            sep="\t",
            index=False,
        )
    if not inter_sgb_df2.empty:
        inter_sgb_df2.to_csv(
            args.output_dir / f"{args.prefix}_classified_hits{suffix2}.tsv",
            sep="\t",
            index=False,
        )

    # --- Analysis Loop ---
    # Loop through each analysis level to compare the two conditions.
    for level in ["global", "within_mouse", "within_replicate", "between_replicates"]:
        logger.info("=" * 20 + f" COMPARISON FOR: {level.upper()} " + "=" * 20)

        # Extract the summary data for the current level.
        detailed1 = results1.get(level, {}).get("detailed_summary", pd.DataFrame())
        aggregated1 = results1.get(level, {}).get("aggregated_summary", pd.DataFrame())
        detailed2 = results2.get(level, {}).get("detailed_summary", pd.DataFrame())
        aggregated2 = results2.get(level, {}).get("aggregated_summary", pd.DataFrame())

        # Save the individual summary tables for each group.
        if not detailed1.empty:
            detailed1.to_csv(
                args.output_dir
                / f"{args.prefix}_{level}_detailed_summary{suffix1}.tsv",
                sep="\t",
                index=False,
            )
        if not aggregated1.empty:
            aggregated1.to_csv(
                args.output_dir
                / f"{args.prefix}_{level}_aggregated_summary{suffix1}.tsv",
                sep="\t",
                index=False,
            )
        if not detailed2.empty:
            detailed2.to_csv(
                args.output_dir
                / f"{args.prefix}_{level}_detailed_summary{suffix2}.tsv",
                sep="\t",
                index=False,
            )
        if not aggregated2.empty:
            aggregated2.to_csv(
                args.output_dir
                / f"{args.prefix}_{level}_aggregated_summary{suffix2}.tsv",
                sep="\t",
                index=False,
            )

        # --- Perform Statistical Comparisons ---
        if not aggregated1.empty and not aggregated2.empty:
            # Merge the aggregated summaries for count-based tests and plotting.
            comp_df = pd.merge(
                aggregated1,
                aggregated2,
                on=["SGB_MAG1", "SGB_MAG2"],
                how="inner",
                suffixes=(suffix1, suffix2),
            )

            # Perform count-based tests (Binomial, Poisson) on the aggregated data.
            comp_df = perform_count_based_tests(comp_df, focus_suffix, baseline_suffix)

            # For stratified levels, perform distributional tests.
            if level in ["within_mouse", "within_replicate", "between_replicates"]:
                dist_test_results = perform_distributional_tests(
                    detailed1, detailed2, level, args.min_samples
                )
                comp_df = pd.merge(
                    comp_df, dist_test_results, on=["SGB_MAG1", "SGB_MAG2"], how="left"
                )

            # Save the final comparison table with all statistical results.
            comp_df.to_csv(
                args.output_dir / f"{args.prefix}_{level}_statistical_comparison.tsv",
                sep="\t",
                index=False,
            )
            logging.info(f"\n--- Statistical Comparison Table ({level}) ---")
            # Generate plots if requested by the user.
            if args.plot:
                plot_data(
                    comp_df,
                    "percentage_interacting",
                    "% of MAG pairs with >=1 BLAST hit",
                    f"Interaction Percentage ({level})",
                    suffix1,
                    suffix2,
                    plot_label1,
                    plot_label2,
                    args.output_dir,
                    args.prefix,
                )
                plot_data(
                    comp_df,
                    "sum_total_number_of_HGTs",
                    "Total number of HGTs",
                    f"Total number of HGTs ({level})",
                    suffix1,
                    suffix2,
                    plot_label1,
                    plot_label2,
                    args.output_dir,
                    args.prefix,
                )
                # The statistical scatter plot can be generated for any level with p-values.
                plot_statistical_results(
                    comp_df,
                    baseline_suffix,
                    focus_suffix,
                    baseline_label,
                    focus_label,
                    level,
                    args.stat_plot_type,
                    args.alpha,
                    args.output_dir,
                    args.prefix,
                )
        else:
            logger.warning(
                f"One or both files had no data for the '{level}' level. Skipping comparison."
            )


if __name__ == "__main__":
    main()
