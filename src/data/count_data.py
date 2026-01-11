#!/usr/bin/env python
import os
from pathlib import Path

def count_images(dataset_root):
    dataset_root = Path(dataset_root)
    data = []
    
    # Iterate over categories
    for category_dir in sorted(dataset_root.iterdir()):
        if not category_dir.is_dir():
            continue
            
        category = category_dir.name
        # Check for nested category folder structure as seen in analysis
        nested_category_dir = category_dir / category
        if nested_category_dir.exists():
            base_dir = nested_category_dir
        else:
            base_dir = category_dir
            
        # Train counts
        train_dir = base_dir / "train" / "good"
        train_count = len(list(train_dir.glob("*.png"))) if train_dir.exists() else 0
        
        data.append({
            "Category": category,
            "Split": "Train",
            "Type": "good",
            "Count": train_count
        })
        
        # Test counts
        test_dir = base_dir / "test"
        if test_dir.exists():
            for defect_dir in sorted(test_dir.iterdir()):
                if not defect_dir.is_dir():
                    continue
                defect_type = defect_dir.name
                count = len(list(defect_dir.glob("*.png")))
                data.append({
                    "Category": category,
                    "Split": "Test",
                    "Type": defect_type,
                    "Count": count
                })

    # Sort data
    data.sort(key=lambda x: (x["Category"], x["Split"], x["Type"]))
    
    # Print markdown table
    print("| Category | Split | Type | Count |")
    print("|---|---|---|---|")
    for row in data:
        print(f"| {row['Category']} | {row['Split']} | {row['Type']} | {row['Count']} |")

if __name__ == "__main__":
    dataset_root = "data/raw"
    try:
        count_images(dataset_root)
    except Exception as e:
        print(f"Error: {e}")
