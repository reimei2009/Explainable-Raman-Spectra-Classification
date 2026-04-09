#!/usr/bin/env python
"""
Module 01 Pipeline Lần 2: Merge Label + Chia Tập + Tạo QC Report
"""

import sys
from pathlib import Path
import pandas as pd

# Add project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.manifest import DatasetManifest
from src.data.labeling import DISEASE_MAPPING
from src.data.split import stratified_patient_split, print_split_summary
from src.data.create_labeling import create_labeling_df, get_disease_label
from src.utils import ensure_dir

def main():
    # Paths
    project_root = Path(__file__).resolve().parent.parent
    data_root = project_root / "Data"
    output_data_dir = project_root / "outputs" / "data"
    output_reports_dir = project_root / "outputs" / "reports"
    
    ensure_dir(output_data_dir)
    ensure_dir(output_reports_dir)
    
    print("\n" + "="*70)
    print("MODULE 01: DATA GOVERNANCE PIPELINE – PHASE 2")
    print("="*70)
    
    # Step 1: Scan samples
    print("\n[Step 1/5] Scanning samples...")
    manifest = DatasetManifest(data_root)
    samples_df = manifest.scan_samples()
    print(f"✓ Found {len(samples_df)} samples")
    
    # Step 2: Load disease labels
    print("\n[Step 2/5] Loading disease labels (Option 1 mapping)...")
    labels_df = create_labeling_df()
    
    # Merge labels
    samples_df = samples_df.merge(
        labels_df[['sample_id', 'disease_label', 'notes']],
        on='sample_id', how='left', suffixes=('', '_label')
    )
    
    # Update disease_label
    mask = samples_df['disease_label_label'].notna()
    samples_df.loc[mask, 'disease_label'] = samples_df.loc[mask, 'disease_label_label']
    samples_df = samples_df.drop('disease_label_label', axis=1)
    
    # Update notes
    samples_df['notes'] = samples_df['notes_label'].fillna('')
    samples_df = samples_df.drop('notes_label', axis=1, errors='ignore')
    
    print(f"✓ Loaded {len(labels_df)} labels")
    
    # Try case-insensitive lookup for missing labels
    for idx, row in samples_df.iterrows():
        if pd.isna(row['disease_label']) or row['disease_label'] is None:
            label, note = get_disease_label(row['sample_id'])
            samples_df.at[idx, 'disease_label'] = label
            if note and not pd.isna(row['notes']):
                samples_df.at[idx, 'notes'] = note
    
    # Step 3: Encode diseases
    print("\n[Step 3/5] Encoding disease labels...")
    manifest.samples = samples_df
    manifest.encode_diseases(DISEASE_MAPPING)
    
    valid_samples = manifest.samples[manifest.samples['disease_code'].notna()].copy()
    excluded_samples = manifest.samples[manifest.samples['disease_code'].isna()].copy()
    
    print(f"✓ Valid samples: {len(valid_samples)}")
    print(f"✓ Excluded samples: {len(excluded_samples)}")
    if len(excluded_samples) > 0:
        for _, row in excluded_samples.iterrows():
            print(f"    - {row['sample_id']}: {row['notes']}")
    
    # Step 4: Split train/val/test
    print("\n[Step 4/5] Splitting train/val/test (patient-level stratified)...")
    train_df, val_df, test_df = stratified_patient_split(valid_samples)
    
    print(print_split_summary(train_df, val_df, test_df))
    
    # Step 5: Save outputs
    print("\n[Step 5/5] Saving outputs...")
    
    # Save manifest v1.0
    manifest_path = output_data_dir / "manifest_v1.0.csv"
    valid_samples.to_csv(manifest_path, index=False)
    print(f"✓ Manifest saved: {manifest_path}")
    
    # Save split summary
    split_summary_path = output_reports_dir / "split_summary.txt"
    with open(split_summary_path, 'w', encoding='utf-8') as f:
        f.write(print_split_summary(train_df, val_df, test_df))
    print(f"✓ Split summary saved: {split_summary_path}")
    
    # Save individual split files
    train_df.to_csv(output_data_dir / "train_samples.csv", index=False)
    val_df.to_csv(output_data_dir / "val_samples.csv", index=False)
    test_df.to_csv(output_data_dir / "test_samples.csv", index=False)
    print(f"✓ Split files saved: train/val/test_samples.csv")
    
    # Save labeling notes
    labeling_notes_path = output_reports_dir / "labeling_notes.txt"
    with open(labeling_notes_path, 'w', encoding='utf-8') as f:
        f.write("LABELING NOTES\n")
        f.write("="*70 + "\n\n")
        f.write("Mapping Strategy: Option 1 (Group + Exclude)\n")
        f.write("- Stone + Sludge Stone -> stone (0)\n")
        f.write("- Polyp -> polyp (1)\n")
        f.write("- Cancer -> cancer (2)\n")
        f.write("- GA, Stone+Polyp -> excluded\n\n")
        f.write(f"Total samples: {len(manifest.samples)}\n")
        f.write(f"Valid samples: {len(valid_samples)}\n")
        f.write(f"Excluded samples: {len(excluded_samples)}\n\n")
        f.write("Excluded samples:\n")
        for _, row in excluded_samples.iterrows():
            f.write(f"  - {row['sample_id']}: {row['notes']}\n")
    print(f"✓ Labeling notes saved: {labeling_notes_path}")
    
    # Summary
    print("\n" + "="*70)
    print("✅ MODULE 01 COMPLETE")
    print("="*70)
    print(f"\nOutputs saved to:")
    print(f"  - {output_data_dir}")
    print(f"  - {output_reports_dir}")
    print(f"\nReady for Module 02: Signal Preprocessing")
    print("="*70 + "\n")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
