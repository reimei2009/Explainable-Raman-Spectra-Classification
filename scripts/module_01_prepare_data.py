#!/usr/bin/env python
"""
Module 01 Pipeline: Chuẩn bị dữ liệu.
Quy trình:
1. Scan samples từ folder
2. Tạo template labeling để người dùng điền
3. (Người dùng điền disease label)
4. Merge label → manifest chính thức
5. Chia train/val/test
6. In tóm tắt split
"""

import sys
from pathlib import Path

# Add project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.manifest import DatasetManifest
from src.data.labeling import DISEASE_MAPPING
from src.data.split import stratified_patient_split, print_split_summary
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
    print("MODULE 01: DATA GOVERNANCE PIPELINE")
    print("="*70)
    
    # Step 1: Scan samples
    print("\n[Step 1/2] Scanning samples from Data/different potential/csv...")
    manifest = DatasetManifest(data_root)
    samples_df = manifest.scan_samples()
    
    print(f"✓ Found {len(samples_df)} samples from {len(samples_df['subject_id'].unique())} bệnh nhân")
    print("\n" + samples_df[['sample_id', 'gender', 'subject_id', 'sample_type', 'available_potentials']].to_string(index=False))
    
    # Step 2: Create labeling template
    template_path = output_data_dir / "labeling_template.csv"
    DatasetManifest.create_labeling_template(samples_df, template_path)
    
    print(f"\n[Step 2/2] ✓ Labeling template created:")
    print(f"   Path: {template_path}")
    print(f"\n⚠️  NEXT ACTION:")
    print(f"   1. Open file: {template_path}")
    print(f"   2. Fill in 'disease_label' column with: stone, polyp, atau cancer")
    print(f"   3. (Optional) Add notes in 'notes' column")
    print(f"   4. Save file")
    print(f"   5. Run this script again: python scripts/module_01_prepare_data.py")
    print(f"\n   Ví dụ:")
    print(f"   sample_id | disease_label | notes")
    print(f"   M78S      | stone         | confirmed by ultrasound")
    print(f"   F60S      | polyp         | suspicious, needs follow-up")
    print(f"   F35S      | cancer        | histology confirmed")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
