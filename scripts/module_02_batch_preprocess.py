"""
MODULE 02 - BATCH PREPROCESSING
Preprocesses all 36 valid training, validation, and test samples.
Generates augmentations for training data only.
Saves processed spectra as numpy arrays.
"""

import os
import sys
import numpy as np
import pandas as pd
from pathlib import Path

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.preprocessing import RamanPreprocessor
from src.augmentation import RamanAugmenter


def main():
    # ===== PATHS =====
    proj_root = Path(__file__).parent.parent
    data_root = proj_root / 'Data' / 'different potential' / 'csv'
    output_dir = proj_root / 'outputs' / 'data'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # === PARAMETERS ===
    TARGET_POINTS = 2048
    N_AUGMENTATIONS = 3  # augmentations per training sample
    
    # ===== LOAD SPLITS =====
    print("\n" + "=" * 80)
    print("MODULE 02 - BATCH PREPROCESSING")
    print("=" * 80)
    
    split_files = {
        'train': output_dir / 'train_samples.csv',
        'val': output_dir / 'val_samples.csv',
        'test': output_dir / 'test_samples.csv'
    }
    
    all_samples = []
    for split, split_file in split_files.items():
        if split_file.exists():
            df = pd.read_csv(split_file)
            df['split'] = split
            all_samples.append(df)
        else:
            print(f"Warning: {split_file} not found")
    
    manifest_df = pd.concat(all_samples, ignore_index=True)
    print(f"\nLoaded {len(manifest_df)} samples from split files")
    print(f"Splits: {', '.join(manifest_df['split'].unique())}\n")
    
    # Initialize preprocessor and augmenter
    np.random.seed(42)  # For reproducible augmentation
    preprocessor = RamanPreprocessor()
    augmenter = RamanAugmenter()
    
    # Storage for processed spectra and augmentations
    processed_data = {
        'train': {'X': [], 'sample_ids': [], 'disease_codes': []},
        'val':   {'X': [], 'sample_ids': [], 'disease_codes': []},
        'test':  {'X': [], 'sample_ids': [], 'disease_codes': []}
    }
    
    augmented_data = {
        'train': {'X': [], 'sample_ids': [], 'aug_indices': [], 'disease_codes': []}
    }
    
    # ===== PROCESS EACH SAMPLE =====
    for split in ['train', 'val', 'test']:
        split_df = manifest_df[manifest_df['split'] == split].reset_index(drop=True)
        print(f"\n{'─' * 80}")
        print(f"Processing {split.upper()} set ({len(split_df)} samples)")
        print(f"{'─' * 80}")
        
        for idx, row in split_df.iterrows():
            sample_id = row['sample_id']
            disease_code = int(row['disease_code'])
            
            # Find all CSV files for this sample across all potentials and replicates
            sample_dir = data_root / sample_id
            if not sample_dir.exists():
                print(f"  ❌ {sample_id:20s} | Directory not found: {sample_dir}")
                continue
            
            # Get all CSV files for this sample
            csv_files = sorted(sample_dir.glob('*.csv'))
            if not csv_files:
                print(f"  ❌ {sample_id:20s} | No CSV files found in {sample_dir}")
                continue
            
            # Process all replicates/potentials and average them
            spectra_list = []
            wn_resampled = None
            for csv_path in csv_files:
                try:
                    df_raw = pd.read_csv(csv_path)
                    wn = df_raw['wavenumber_cm1'].values
                    intensity = df_raw['intensity'].values
                    
                    # Preprocess
                    wn_resampled_cur, intensity_proc = preprocessor.preprocess_spectrum(
                        wavenumber=wn,
                        intensity=intensity,
                        crop_range=None,
                        target_points=TARGET_POINTS,
                        baseline_method='airpls',
                        smooth=True
                    )
                    spectra_list.append(intensity_proc)
                    if wn_resampled is None:
                        wn_resampled = wn_resampled_cur
                except Exception as e:
                    print(f"    Warning: Failed to process {csv_path.name}: {str(e)}")
                    continue
            
            if not spectra_list:
                print(f"  ❌ {sample_id:20s} | Failed to process any spectra")
                continue
            
            # Average all replicates/potentials for the sample
            intensity_avg = np.mean(spectra_list, axis=0)
            
            # Store processed spectrum
            processed_data[split]['X'].append(intensity_avg)
            processed_data[split]['sample_ids'].append(sample_id)
            processed_data[split]['disease_codes'].append(disease_code)
            
            # Generate augmentations for training data
            if split == 'train':
                for aug_idx in range(N_AUGMENTATIONS):
                    aug_items = augmenter.augment(
                        wavenumber=wn_resampled,
                        intensity=intensity_avg,
                        n_aug=1,
                    )
                    _, intensity_aug = aug_items[0]
                    augmented_data['train']['X'].append(intensity_aug)
                    augmented_data['train']['sample_ids'].append(sample_id)
                    augmented_data['train']['aug_indices'].append(aug_idx)
                    augmented_data['train']['disease_codes'].append(disease_code)
            
            n_replicates = len(csv_files)
            print(f"  ✅ {sample_id:20s} | {n_replicates:2d} replicates averaged | "
                  f"Disease: {disease_code}")
    
    # ===== CONVERT TO NUMPY ARRAYS =====
    print(f"\n{'═' * 80}")
    print("Finalizing arrays...")
    print(f"{'═' * 80}\n")
    
    splits_info = {}
    for split in ['train', 'val', 'test']:
        if processed_data[split]['X']:
            X = np.array(processed_data[split]['X'], dtype=np.float32)
            sample_ids = processed_data[split]['sample_ids']
            disease_codes = np.array(processed_data[split]['disease_codes'], dtype=np.int32)
            
            splits_info[split] = {
                'X': X,
                'sample_ids': sample_ids,
                'disease_codes': disease_codes,
                'n_samples': X.shape[0],
                'n_features': X.shape[1]
            }
            
            print(f"{split.upper():5s} set: {X.shape[0]:3d} samples × {X.shape[1]:4d} features")
    
    # Add augmented training data
    if augmented_data['train']['X']:
        X_aug = np.array(augmented_data['train']['X'], dtype=np.float32)
        print(f"{'AUG':5s} set: {X_aug.shape[0]:3d} augmented samples × {X_aug.shape[1]:4d} features")
    
    # ===== SAVE ARRAYS =====
    print(f"\n{'─' * 80}")
    print("Saving processed arrays...")
    print(f"{'─' * 80}\n")
    
    for split, info in splits_info.items():
        X_path = output_dir / f"X_{split}.npy"
        np.save(X_path, info['X'])
        print(f"  ✅ {X_path.name:25s} | Shape: {info['X'].shape}")
    
    if augmented_data['train']['X']:
        X_aug = np.array(augmented_data['train']['X'], dtype=np.float32)
        X_aug_path = output_dir / "X_train_augmented.npy"
        np.save(X_aug_path, X_aug)
        print(f"  ✅ {X_aug_path.name:25s} | Shape: {X_aug.shape}")
    
    # Save metadata
    metadata_dict = {
        'train': {'sample_ids': processed_data['train']['sample_ids'], 'disease_codes': processed_data['train']['disease_codes']},
        'val':   {'sample_ids': processed_data['val']['sample_ids'], 'disease_codes': processed_data['val']['disease_codes']},
        'test':  {'sample_ids': processed_data['test']['sample_ids'], 'disease_codes': processed_data['test']['disease_codes']},
        'aug_train': {'sample_ids': augmented_data['train']['sample_ids'], 'aug_indices': augmented_data['train']['aug_indices'], 'disease_codes': augmented_data['train']['disease_codes']}
    }
    
    # Save as numpy archive
    metadata_path = output_dir / "metadata.npz"
    metadata_arrays = {}
    for split_name, split_meta in metadata_dict.items():
        for field_name, field_values in split_meta.items():
            key = f"{split_name}_{field_name}"
            if field_name == "sample_ids":
                metadata_arrays[key] = np.array([str(x) for x in field_values], dtype=object)
            else:
                metadata_arrays[key] = np.array(field_values)
    np.savez(metadata_path, **metadata_arrays)
    print(f"  ✅ {metadata_path.name:25s} | Metadata saved")
    
    # ===== SUMMARY REPORT =====
    print(f"\n{'═' * 80}")
    print("BATCH PREPROCESSING SUMMARY")
    print(f"{'═' * 80}\n")
    
    print("Dataset Composition:")
    for split, info in splits_info.items():
        disease_counts = {}
        for dc in info['disease_codes']:
            disease_counts[dc] = disease_counts.get(dc, 0) + 1
        
        disease_names = {0: 'stone', 1: 'polyp', 2: 'cancer'}
        disease_str = " | ".join([f"{disease_names[k]}: {v}" for k, v in sorted(disease_counts.items())])
        print(f"  {split.upper():5s} ({info['n_samples']:2d}): {disease_str}")
    
    print(f"\nProcessing Parameters:")
    print(f"  Target points:     {TARGET_POINTS}")
    print(f"  Augmentations:     {N_AUGMENTATIONS} per training sample")
    print(f"  Baseline method:   airPLS (λ=1e5)")
    print(f"  Normalization:     min-max [0, 1]")
    
    total_train_samples = len(processed_data['train']['sample_ids'])
    total_train_augmented = len(augmented_data['train']['sample_ids'])
    print(f"\nTraining Data:")
    print(f"  Original samples:  {total_train_samples}")
    print(f"  Augmented samples: {total_train_augmented}")
    print(f"  Total in memory:   {total_train_samples + total_train_augmented}")
    
    print(f"\nOutput Files:")
    print(f"  ✅ X_train.npy (shape: {splits_info['train']['X'].shape})")
    print(f"  ✅ X_val.npy   (shape: {splits_info['val']['X'].shape})")
    print(f"  ✅ X_test.npy  (shape: {splits_info['test']['X'].shape})")
    if augmented_data['train']['X']:
        print(f"  ✅ X_train_augmented.npy (shape: {X_aug.shape})")
    print(f"  ✅ metadata.npz (train/val/test sample IDs and disease codes)")
    
    print(f"\n{'═' * 80}")
    print("✅ BATCH PREPROCESSING COMPLETE")
    print(f"{'═' * 80}\n")


if __name__ == '__main__':
    main()
