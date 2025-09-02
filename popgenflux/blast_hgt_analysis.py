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

EPS = 1e-12

# Set up a global logger for the script. This allows logging from any function.
logger = logging.getLogger(__name__)


def _map_seqids_to_mag_ids(
    seqids: pd.Series,
    contig_map_df: pd.DataFrame,
) -> pd.Series:
    """Map BLAST sequence IDs (gene IDs) to MAG IDs via contig→MAG map.

    Expected gene ID shape: contigID_geneID (contig and gene separated by an underscore).
    Mapping uses the prefix before the last underscore as the contig ID.

    Args:
        seqids: pandas Series containing BLAST sequence IDs (gene IDs)
        contig_map_df: DataFrame with columns 'contig_id' and 'mag_id' for mapping

    Returns:
        pandas Series of MAG IDs corresponding to input sequence IDs

    Raises:
        ValueError: If contig_map_df is missing required columns
        SystemExit: If any sequence ID cannot be mapped to a MAG or if
                   duplicate contig mappings are found
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
    """
    Generate detailed and aggregated summaries for all analysis levels.

    This function creates comprehensive summaries of HGT (horizontal gene transfer)
    interactions between SGB (Species Genome Bin) pairs across four hierarchical
    analysis levels: global, within_mouse, within_replicate, and between_replicates.

    The function first builds a complete "scaffold" of all possible SGB pair contexts,
    calculates theoretical denominators (total possible MAG-MAG interactions), and
    then merges the observed hit counts to compute interaction percentages.

    Args:
        inter_sgb_df: DataFrame containing classified inter-SGB hits with columns:
            - SGB1_canonical, SGB2_canonical: Canonical SGB pair identifiers
            - canonical_pair_key: Unique MAG pair identifier
            - hgt_category: Classification level (within_mouse, within_replicate, between_replicates)
            - sample_id1, sample_id2: Sample identifiers for each hit
            - replicate1, replicate2: Replicate identifiers for each hit
        context_map_df: DataFrame mapping MAGs to experimental context with columns:
            - MAG_ID: MAG identifier
            - SGB: Species Genome Bin identifier
            - sample_id: Sample/mouse identifier
            - replicate: Experimental replicate identifier

    Returns:
        dict: Nested dictionary with structure:
            {
                'level_name': {
                    'detailed_summary': DataFrame with context-stratified results,
                    'aggregated_summary': DataFrame with SGB-pair level aggregates
                }
            }
            where level_name is one of: 'global', 'within_mouse', 'within_replicate', 'between_replicates'

    Note:
        - Global level has no detailed_summary (returns empty DataFrame)
        - Denominators represent theoretical maximum MAG-MAG interactions
        - Percentages calculated as (observed interactions / theoretical maximum) * 100
    """
    # Initialize results dictionary to store summaries for each analysis level
    results = {}

    # Create master list of all unique SGB identifiers from the context mapping
    all_sgbs = sorted(context_map_df["SGB"].dropna().unique())

    # Generate all possible SGB pairs for global-level analysis scaffold
    # This ensures we capture all potential interactions, even those with zero hits
    master_sgb_pairs_df = pd.DataFrame(
        list(combinations(all_sgbs, 2)), columns=["SGB_MAG1", "SGB_MAG2"]
    )

    # Define analysis levels and their corresponding data subsets
    # Each level represents a different biological/experimental scale
    analysis_levels = {
        "global": inter_sgb_df,  # All inter-SGB hits regardless of context
        "within_mouse": inter_sgb_df[
            inter_sgb_df["hgt_category"] == "within_mouse"
        ].copy(),  # Hits within the same individual mouse
        "within_replicate": inter_sgb_df[
            inter_sgb_df["hgt_category"] == "within_replicate"
        ].copy(),  # Hits within same replicate but different mice
        "between_replicates": inter_sgb_df[
            inter_sgb_df["hgt_category"] == "between_replicates"
        ].copy(),  # Hits between different experimental replicates
    }

    # Check for missing data that could affect context-specific analyses
    num_missing_samples = context_map_df["sample_id"].isna().sum()
    num_missing_replicates = context_map_df["replicate"].isna().sum()
    if num_missing_samples:
        logger.warning(
            f"{num_missing_samples} rows have missing sample_id; they are excluded from context scaffolds."
        )
    if num_missing_replicates:
        logger.warning(
            f"{num_missing_replicates} rows have missing replicate; they are excluded from replicate-based scaffolds."
        )

    # Process each analysis level separately
    for level, hits_subset in analysis_levels.items():
        logger.info(f"Generating summaries for '{level}' level...")

        # Initialize empty scaffold - will be populated based on analysis level
        scaffold_df = pd.DataFrame()

        # === LEVEL-SPECIFIC SCAFFOLD GENERATION ===
        # Each level requires different denominator calculations based on context

        if level == "within_mouse":
            # For within-mouse analysis: calculate possible MAG pairs within each mouse
            scaffold_rows = []

            # Count unique MAGs per SGB per mouse (sample_id)
            mag_counts_per_mouse = (
                context_map_df.dropna(subset=["sample_id"])
                .groupby(["sample_id", "SGB"])["MAG_ID"]
                .nunique()
                .unstack(
                    fill_value=0
                )  # Pivot: rows=sample_id, cols=SGB, values=MAG_count
            )

            if not mag_counts_per_mouse.empty:
                sgb_columns = list(mag_counts_per_mouse.columns)

                # For each SGB pair, calculate possible MAG interactions per mouse
                for sgb_first, sgb_second in combinations(sgb_columns, 2):
                    # Denominator = MAGs_in_SGB1 × MAGs_in_SGB2 for each mouse
                    possible_pairs_per_mouse = (
                        mag_counts_per_mouse[sgb_first]
                        * mag_counts_per_mouse[sgb_second]
                    ).rename("total_possible_mag_pairs")

                    # Convert to DataFrame and add SGB pair information
                    mouse_context_df = possible_pairs_per_mouse.reset_index().rename(
                        columns={"sample_id": "context_id"}
                    )
                    mouse_context_df["SGB_MAG1"], mouse_context_df["SGB_MAG2"] = (
                        sgb_first,
                        sgb_second,
                    )
                    scaffold_rows.append(mouse_context_df)

            # Combine all scaffold rows or create empty DataFrame with expected columns
            scaffold_df = (
                pd.concat(scaffold_rows, ignore_index=True)
                if scaffold_rows
                else pd.DataFrame(
                    columns=[
                        "context_id",
                        "SGB_MAG1",
                        "SGB_MAG2",
                        "total_possible_mag_pairs",
                    ]
                )
            )

        elif level == "within_replicate":
            # For within-replicate analysis: calculate cross-mouse MAG pairs within replicates
            scaffold_rows = []

            # Process each replicate separately
            for replicate_id, replicate_df in context_map_df.dropna(
                subset=["replicate"]
            ).groupby("replicate"):
                # Create contingency table: rows=sample_id, cols=SGB, values=MAG_count
                mag_contingency_table = (
                    replicate_df.groupby(["sample_id", "SGB"])["MAG_ID"]
                    .nunique()
                    .unstack(fill_value=0)
                )

                if mag_contingency_table.empty:
                    continue

                sgb_columns = list(mag_contingency_table.columns)
                # Sum MAGs across all mice in this replicate for each SGB
                total_mags_per_sgb = mag_contingency_table.sum(axis=0)

                # For each SGB pair, calculate cross-mouse interactions within replicate
                for sgb_first, sgb_second in combinations(sgb_columns, 2):
                    # Diagonal term: sum of (MAGs_in_SGB1 × MAGs_in_SGB2) within each mouse
                    # This represents within-mouse interactions that should be excluded
                    within_mouse_pairs = (
                        mag_contingency_table[sgb_first]
                        * mag_contingency_table[sgb_second]
                    ).sum()

                    # Total possible pairs minus within-mouse pairs = between-mouse pairs
                    between_mouse_pairs = int(
                        total_mags_per_sgb.get(sgb_first, 0)
                        * total_mags_per_sgb.get(sgb_second, 0)
                        - within_mouse_pairs
                    )

                    scaffold_rows.append(
                        {
                            "context_id": replicate_id,
                            "SGB_MAG1": sgb_first,
                            "SGB_MAG2": sgb_second,
                            "total_possible_mag_pairs": between_mouse_pairs,
                        }
                    )

            scaffold_df = pd.DataFrame(scaffold_rows)

        elif level == "between_replicates":
            # For between-replicates analysis: calculate MAG pairs across different replicates
            scaffold_rows = []

            # Count MAGs per SGB per replicate
            mag_counts_per_replicate = (
                context_map_df.dropna(subset=["replicate"])
                .groupby(["replicate", "SGB"])["MAG_ID"]
                .nunique()
                .unstack(fill_value=0)
            )

            if not mag_counts_per_replicate.empty:
                replicate_list = list(mag_counts_per_replicate.index)
                sgb_columns = list(mag_counts_per_replicate.columns)

                # For each pair of replicates
                for replicate_first, replicate_second in combinations(
                    replicate_list, 2
                ):
                    replicate_first_counts = mag_counts_per_replicate.loc[
                        replicate_first
                    ]
                    replicate_second_counts = mag_counts_per_replicate.loc[
                        replicate_second
                    ]

                    # For each SGB pair, calculate cross-replicate interactions
                    for sgb_first, sgb_second in combinations(sgb_columns, 2):
                        # Bidirectional interactions between replicates:
                        # (SGB1_rep1 × SGB2_rep2) + (SGB1_rep2 × SGB2_rep1)
                        cross_replicate_pairs = int(
                            replicate_first_counts.get(sgb_first, 0)
                            * replicate_second_counts.get(sgb_second, 0)
                            + replicate_second_counts.get(sgb_first, 0)
                            * replicate_first_counts.get(sgb_second, 0)
                        )

                        scaffold_rows.append(
                            {
                                "context_id": tuple(
                                    sorted((replicate_first, replicate_second))
                                ),
                                "SGB_MAG1": sgb_first,
                                "SGB_MAG2": sgb_second,
                                "total_possible_mag_pairs": cross_replicate_pairs,
                            }
                        )

            scaffold_df = pd.DataFrame(scaffold_rows)

        # === OBSERVED HITS PROCESSING ===
        # For non-global levels, process observed hits and merge with scaffold
        if level != "global":
            # Define grouping columns for aggregating observed hits by context
            level_grouping_columns = {
                "within_mouse": ["sample_id1", "SGB1_canonical", "SGB2_canonical"],
                "within_replicate": ["replicate1", "SGB1_canonical", "SGB2_canonical"],
                "between_replicates": [
                    "replicate_pair",
                    "SGB1_canonical",
                    "SGB2_canonical",
                ],
            }

            # Special handling for between_replicates: create replicate pair identifiers
            if level == "between_replicates":
                hits_subset["replicate_pair"] = hits_subset.apply(
                    lambda row: tuple(sorted((row["replicate1"], row["replicate2"]))),
                    axis=1,
                )

            # Aggregate observed hits by context and SGB pair
            observed_hits_summary = (
                hits_subset.groupby(level_grouping_columns[level])
                .agg(
                    interacting_mag_pairs=(
                        "canonical_pair_key",
                        "nunique",
                    ),
                    total_number_of_HGTs=(
                        "canonical_pair_key",
                        "size",
                    ),
                )
                .reset_index()
            )

            # Standardize column names for merging with scaffold
            context_column_mapping = {
                "within_mouse": "sample_id1",
                "within_replicate": "replicate1",
                "between_replicates": "replicate_pair",
            }
            observed_hits_summary.rename(
                columns={
                    context_column_mapping[level]: "context_id",
                    "SGB1_canonical": "SGB_MAG1",
                    "SGB2_canonical": "SGB_MAG2",
                },
                inplace=True,
            )

            # Merge scaffold (all possible contexts/pairs) with observed hits
            # Left join ensures we keep all possible pairs, even those with zero hits
            detailed_summary = pd.merge(
                scaffold_df,
                observed_hits_summary,
                on=["context_id", "SGB_MAG1", "SGB_MAG2"],
                how="left",
            )

            # Fill missing values (no hits) with zeros and ensure integer types
            detailed_summary[["interacting_mag_pairs", "total_number_of_HGTs"]] = (
                detailed_summary[["interacting_mag_pairs", "total_number_of_HGTs"]]
                .fillna(0)
                .astype(int)
            )

            # Calculate percentage of possible MAG pairs that actually interact
            detailed_summary["percentage_interacting"] = np.where(
                detailed_summary["total_possible_mag_pairs"] > 0,
                100.0
                * detailed_summary["interacting_mag_pairs"]
                / detailed_summary["total_possible_mag_pairs"],
                np.nan,  # Undefined when no possible pairs exist
            )

        # === GLOBAL LEVEL PROCESSING ===
        # Global level uses different logic - no context stratification
        if level == "global":
            # Count total unique MAGs per SGB across all contexts
            total_mags_per_sgb = context_map_df.groupby("SGB")["MAG_ID"].nunique()

            # Calculate global denominators for all SGB pairs
            global_denominators = master_sgb_pairs_df.copy()
            global_denominators["sum_total_possible_mag_pairs"] = (
                global_denominators.apply(
                    lambda row: int(
                        total_mags_per_sgb.get(row["SGB_MAG1"], 0)
                        * total_mags_per_sgb.get(row["SGB_MAG2"], 0)
                    ),
                    axis=1,
                )
            )

            # Aggregate all observed hits globally by SGB pair
            global_hits_aggregated = (
                hits_subset.groupby(["SGB1_canonical", "SGB2_canonical"])
                .agg(
                    sum_interacting_mag_pairs=("canonical_pair_key", "nunique"),
                    sum_total_number_of_HGTs=("canonical_pair_key", "size"),
                )
                .reset_index()
                .rename(
                    columns={"SGB1_canonical": "SGB_MAG1", "SGB2_canonical": "SGB_MAG2"}
                )
            )

            # Merge denominators with observed hits (left join keeps all SGB pairs)
            aggregated_summary = global_denominators.merge(
                global_hits_aggregated, on=["SGB_MAG1", "SGB_MAG2"], how="left"
            ).fillna(0)

            # Ensure correct data types for numeric columns
            numeric_columns_to_cast = {
                "sum_total_possible_mag_pairs": int,
                "sum_interacting_mag_pairs": int,
                "sum_total_number_of_HGTs": int,
            }
            aggregated_summary = aggregated_summary.astype(
                {
                    column_name: data_type
                    for column_name, data_type in numeric_columns_to_cast.items()
                    if column_name in aggregated_summary.columns
                }
            )

            # Global level has no context-specific detailed summary
            detailed_summary = pd.DataFrame()

        else:
            # For non-global levels: create aggregated summary from detailed summary
            # Aggregate across all contexts to get SGB-pair level summaries
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

        # Store both detailed and aggregated summaries for this level
        results[level] = {
            "detailed_summary": detailed_summary,
            "aggregated_summary": aggregated_summary,
        }

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
    # Note: These categories are mutually exclusive by design:
    # - "within_mouse": Same sample_id (intra-individual)
    # - "within_replicate": Same replicate but different sample_id (inter-individual, intra-replicate)
    # - "between_replicates": Different replicate (inter-replicate)
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
    """Compute statistical significance tests comparing hit counts between two conditions.

    Performs both Binomial and Poisson tests to assess whether the focus condition
    has significantly more or fewer hits than expected based on the baseline condition.

    Args:
        comp_df: Merged DataFrame containing aggregated summaries from both conditions
            with suffixed column names (e.g., 'sum_interacting_mag_pairs_GroupA_T0')
        focus_suffix: Suffix identifying the focus condition columns (e.g., '_GroupA_T0')
        baseline_suffix: Suffix identifying the baseline condition columns (e.g., '_GroupB_T0')

    Returns:
        DataFrame with added statistical test columns:
            - p_high_binomial/p_high_poisson: P-value for enrichment (focus > baseline rate)
            - p_low_binomial/p_low_poisson: P-value for depletion (focus < baseline rate)
            - effect_direction: 'enriched', 'depleted', or 'no_change'

    Note:
        Uses a pseudocount approach when baseline has zero hits to avoid infinite significance.
        Both tests assess the same hypothesis but with different distributional assumptions.
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

        # Implement pseudocount logic if baseline (control) hits are zero.
        # This prevents a rate of 0, which would make any non-zero focus count
        # infinitely significant. Instead, we use a very small positive rate.
        if baseline_hits == 0:
            baseline_rate = (1 / baseline_trials) - EPS
            baseline_rate = max(EPS, baseline_rate)  # Ensure rate is not negative
        else:
            baseline_rate = baseline_hits / baseline_trials

        # Calculate expected number of hits in focus condition based on baseline rate
        poisson_mean = focus_trials * baseline_rate

        # Calculate one-tailed p-values for both directions
        # sf() = survival function = P(X > k) = 1 - P(X <= k)
        # cdf() = cumulative distribution function = P(X <= k)
        p_high_binomial = binom.sf(
            focus_hits - 1, focus_trials, baseline_rate
        )  # P(X >= focus_hits)
        p_low_binomial = binom.cdf(
            focus_hits, focus_trials, baseline_rate
        )  # P(X <= focus_hits)
        p_high_poisson = poisson.sf(focus_hits - 1, poisson_mean)  # P(X >= focus_hits)
        p_low_poisson = poisson.cdf(focus_hits, poisson_mean)  # P(X <= focus_hits)

        # Determine effect direction based on comparison to expected value
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
    """Main entry point for HGT comparison analysis.

    Workflow overview:
    1. Parse command-line arguments for input files and parameters
    2. Load and validate input data (BLAST files, mapping files)
    3. Process each BLAST file: filter, map to context, classify hits
    4. Calculate denominators (possible MAG pairs) for statistical comparison
    5. For each analysis level: merge data, calculate statistics, generate plots
    6. Save all results to output directory

    The analysis compares HGT patterns between two experimental conditions
    across multiple biological scales (global, within-mouse, etc.).
    """
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
    # Configure logging and create output directory
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

    # --- Data Loading Phase ---
    # Load the mapping file, which connects MAGs to SGBs and experimental context.
    logger.info("Loading mapping and contig files...")
    required_cols = [
        "MAG_ID",
        "SGB",
        "sample_id",
        "replicate",
        # "subjectID",
        "time",
        "group",
    ]
    map_df = pd.read_csv(args.mapping_file, usecols=required_cols, sep="\t")

    # Load contig→MAG mapping (required for sequence ID to MAG ID conversion)
    contig_map_df = pd.read_csv(args.contig_map, sep="\t")

    # --- Core Processing Phase ---
    # Process each BLAST file: filter hits, map to context, classify by analysis level
    logger.info("Processing BLAST files...")
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

    # --- Comparison Setup Phase ---
    # Create unique identifiers and determine focus vs baseline conditions
    logger.info("Setting up comparison between conditions...")
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
    logger.info(
        f"Focus condition: {focus_label}. Baseline condition: {baseline_label}."
    )
    for level in ["global", "within_mouse", "within_replicate", "between_replicates"]:
        logger.info("=" * 20 + f" COMPARISON FOR: {level.upper()} " + "=" * 20)

        detailed1 = results1.get(level, {}).get("detailed_summary", pd.DataFrame())
        detailed2 = results2.get(level, {}).get("detailed_summary", pd.DataFrame())
        aggregated1 = results1.get(level, {}).get("aggregated_summary", pd.DataFrame())
        aggregated2 = results2.get(level, {}).get("aggregated_summary", pd.DataFrame())

        # Save individual summary tables, filtering detailed tables to only show contexts with hits
        summaries_to_save = [
            (detailed1, "detailed_summary", suffix1, "interacting_mag_pairs"),
            (detailed2, "detailed_summary", suffix2, "interacting_mag_pairs"),
            (aggregated1, "aggregated_summary", suffix1, None),
            (aggregated2, "aggregated_summary", suffix2, None),
        ]
        for df, summary_type, suffix, filter_col in summaries_to_save:
            if not df.empty:
                # Only save the detailed summary if there are interacting MAG pairs
                df_to_save = df[df[filter_col] > 0] if filter_col else df
                out_path = (
                    args.output_dir
                    / f"{args.prefix}_{level}_{summary_type}{suffix}.tsv"
                )
                df_to_save.to_csv(out_path, sep="\t", index=False)

        # Combine summaries for comparison
        comp_df = pd.merge(
            aggregated1,
            aggregated2,
            on=["SGB_MAG1", "SGB_MAG2"],
            how="outer",
            suffixes=(suffix1, suffix2),
        )
        comp_df.fillna(0, inplace=True)
        for root in [
            "sum_interacting_mag_pairs",
            "sum_total_number_of_HGTs",
            "sum_total_possible_mag_pairs",
        ]:
            for suf in (suffix1, suffix2):
                col = f"{root}{suf}"
                if col in comp_df.columns:
                    comp_df[col] = comp_df[col].astype(int)

        # Calculate percentages after the merge is complete and NaNs are filled
        denom_col1, denom_col2 = (
            f"sum_total_possible_mag_pairs{suffix1}",
            f"sum_total_possible_mag_pairs{suffix2}",
        )
        num_col1, num_col2 = (
            f"sum_interacting_mag_pairs{suffix1}",
            f"sum_interacting_mag_pairs{suffix2}",
        )
        comp_df[f"percentage_interacting{suffix1}"] = np.where(
            comp_df[denom_col1] > 0, (comp_df[num_col1] / comp_df[denom_col1]) * 100, 0
        )
        comp_df[f"percentage_interacting{suffix2}"] = np.where(
            comp_df[denom_col2] > 0, (comp_df[num_col2] / comp_df[denom_col2]) * 100, 0
        )

        initial_rows = len(comp_df)
        comp_df = comp_df[(comp_df[denom_col1] + comp_df[denom_col2]) > 0]

        logger.info(
            f"Filtered out {initial_rows - len(comp_df)} rows with no activity in either group."
        )
        if comp_df.empty:
            logger.warning(
                f"No SGB pairs with activity to compare for level '{level}'. Skipping."
            )
            continue

        # Perform count-based tests (Binomial, Poisson) on the aggregated data.
        comp_df = perform_count_based_tests(comp_df, focus_suffix, baseline_suffix)

        # For stratified levels, perform distributional tests.
        if level in ["within_mouse", "within_replicate", "between_replicates"]:
            # Keep only contexts with at least one interacting MAG pair
            detailed1 = detailed1[detailed1["interacting_mag_pairs"] > 0].copy()
            detailed2 = detailed2[detailed2["interacting_mag_pairs"] > 0].copy()

            dist_test_results = perform_distributional_tests(
                detailed1, detailed2, level, args.min_samples
            )
            if not dist_test_results.empty:
                comp_df = pd.merge(
                    comp_df, dist_test_results, on=["SGB_MAG1", "SGB_MAG2"], how="left"
                )

        # Save the final comparison table with all statistical results.
        comp_df.to_csv(
            args.output_dir / f"{args.prefix}_{level}_statistical_comparison.tsv",
            sep="\t",
            index=False,
        )

        # --- Visualization Phase (optional) ---
        # Generate plots if requested by the user to visualize the comparisons
        if args.plot:
            # Plot 1: Interaction percentages between conditions
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
            # Plot 2: Total HGT counts between conditions
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
            # Plot 3: Statistical significance scatter plot with color coding
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


if __name__ == "__main__":
    main()
