# This repository contains code for performing **Linear Discriminant Analysis (LDA)** on a multi-group dataset, reducing dimensionality to the first two components, and then generating an 8-grp and a 3-grp LDA model and computing **Bhattacharyya distances** between group distributions. The distances are further visualized as a **network diagram** to highlight relationships between groups.

The raw data files referenced in this code are not included in this repository, as they contain patient-identifiable information; releasing such data would contravene the General Data Protection Regulation (GDPR, Regulation (EU) 2016/679) and equivalent data protection legislation. However, synthetic datasets replicating the structure of the original data can be provided for methodological validation and reproducibility upon reasonable request.

## Features
- Preprocesses data with **SMOTE** to handle class imbalance.
- Scales features using **StandardScaler**.
- Fits an **8-group LDA model** with Stratified cross-validation.
- in the other 3-group LDA model, it fits a 3-group data to the model
- Projects data into **LDA space (components 1 & 2)**.
- Computes **pairwise Bhattacharyya distances** between group distributions.
- Exports distance matrices as CSV.
- Generates a **network graph** where edge thickness corresponds to similarity.

## Project Structure
+-- lda_8grps_victory.py and lda_3grps_victory.py # Main scripts
+-- README.md # This file
+-- data/
  +-- appropriate data (8 group and 3 group) for analysis
+-- outputs/
  +-- network-plots presented as figure4 in the manuscript.
- 


References
-Bhattacharyya, A. (1943). On a measure of divergence between two statistical populations defined by their probability distributions.
-Fisher, R.A. (1936). The use of multiple measurements in taxonomic problems.

License
This project is licensed under the MIT License.