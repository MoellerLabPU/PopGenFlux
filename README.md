# PopGenFlux

Population genetics utilities and CLI tools. Includes an HGT (horizontal gene transfer) analysis workflow comparing BLAST outputs across experimental groups.

## Install

Create the conda environment:
```
# Download the repository
git clone https://github.com/MoellerLabPU/PopGenFlux
cd PopGenFlux
conda env create -f environment.yaml
conda activate popgenflux
```

## Usage

After installing, a CLI is available:
```
popgenflux-hgt --help
```

Example:
```
popgenflux-hgt \
  --blast1 /abs/path/blast_groupA_T0.tsv --group1 GroupA --timepoint1 T0 \
  --blast2 /abs/path/blast_groupB_T0.tsv --group2 GroupB --timepoint2 T0 \
  --focus-group GroupA \
  --mapping_file /abs/path/mag_to_sgb_mapping.tsv \
  --contig-map /abs/path/contig_to_mag.tsv \
  --pident 100 --evalue 0 --length 500 \
  --plot --stat_plot_type poisson --alpha 0.05 \
  --output-dir /abs/path/hgt_analysis_results --prefix hgt_cmp --min-samples 2
```
