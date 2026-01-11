# Data Analysis Report

## Executive Summary
The `data` folder contains a structured dataset suitable for anomaly detection tasks, specifically formatted for the `patchcore_training.py` script. It includes raw image data organized by category and downloaded zip archives.

## Directory Structure

### `data/downloads`
Contains zip archives for each category. These likely serve as the source for the extracted data in `data/raw`.
-   `engine_wiring.zip`
-   `pipe_clip.zip`
-   `pipe_staple.zip`
-   `tank_screw.zip`
-   `underbody_pipes.zip`
-   `underbody_screw.zip`

### `data/raw`
Contains extracted and organized image data for each category.
-   **Categories**: `engine_wiring`, `pipe_clip`, `pipe_staple`, `tank_screw`, `underbody_pipes`, `underbody_screw`.

#### Category Structure (e.g., `engine_wiring`)
Each category folder follows a standard MVTec AD / AutoVI structure:
-   `train/`: Contains "good" images for training.
-   `test/`: Contains subdirectories for "good" and various defect types (e.g., "broken", "missing").
-   `ground_truth/`: Contains pixel-level annotation masks for defects.
-   `defects_config.json`: Metadata about defect types.
-   `defect_example.png`: Visualization of defects.
-   `readme.md` & `license.txt`: Documentation and licensing.

## Data Compatibility
The data structure is fully compatible with `patchcore_training.py`.
-   **Dataset Class**: `AutoVIAnomalyDataset` in the script expects `dataset_root/category/category/train/good` and `dataset_root/category/category/test`.
-   **Note**: The script expects a double nesting of the category name (e.g., `data/raw/engine_wiring/engine_wiring`). *Verification needed: The current listing shows `data/raw/engine_wiring/engine_wiring` exists, confirming this specific nesting requirement.*

## Recommendations
1.  **Training**: You can immediately run `patchcore_training.py` using `data/raw` as the dataset root.
    ```bash
    python patchcore_training.py --dataset-root data/raw --categories engine_wiring
    ```
2.  **Data Management**: Keep `data/downloads` as a backup. If space is an issue, they can be removed since the data is extracted in `data/raw`.
3.  **New Data**: If adding new categories, ensure they follow the nested `category/category/{train,test}` structure to work with the existing script.

## Data Counts
The following table details the number of images available for each category, split, and defect type.

| Category | Split | Type | Count |
|---|---|---|---|
| engine_wiring | Test | blue_hoop | 5 |
| engine_wiring | Test | cardboard | 5 |
| engine_wiring | Test | fastening | 277 |
| engine_wiring | Test | good | 285 |
| engine_wiring | Test | multiple | 33 |
| engine_wiring | Test | obstruction | 2 |
| engine_wiring | Train | good | 285 |
| pipe_clip | Test | good | 195 |
| pipe_clip | Test | operator | 1 |
| pipe_clip | Test | unclipped | 141 |
| pipe_clip | Train | good | 195 |
| pipe_staple | Test | good | 188 |
| pipe_staple | Test | missing | 117 |
| pipe_staple | Train | good | 191 |
| tank_screw | Test | good | 318 |
| tank_screw | Test | missing | 95 |
| tank_screw | Train | good | 318 |
| underbody_pipes | Test | good | 161 |
| underbody_pipes | Test | multiple | 1 |
| underbody_pipes | Test | obstruction | 180 |
| underbody_pipes | Test | operator | 3 |
| underbody_pipes | Train | good | 161 |
| underbody_screw | Test | good | 374 |
| underbody_screw | Test | missing | 18 |
| underbody_screw | Train | good | 373 |
