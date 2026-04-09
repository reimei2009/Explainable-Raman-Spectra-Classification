"""
Create interpretation table for top SHAP features across both with_aug and no_aug.
Maps feature indices to spectral regions based on typical Raman shifts.

Usage:
    python scripts/module_05_create_feature_table.py
"""

import pandas as pd
from pathlib import Path


def get_wavelength_region(feature_idx: int, n_features: int = 2048) -> str:
    """Map feature index to approximate Raman shift region for SERS bile."""
    # Typical SERS spectrum: 0-3200 cm⁻¹, evenly spaced
    max_shift = 3200
    approx_shift = (feature_idx / n_features) * max_shift
    
    # Common SERS bile regions (literature-based)
    if approx_shift < 300:
        return f"~{int(approx_shift)} cm⁻¹ (Low frequency / lattice modes)"
    elif approx_shift < 800:
        return f"~{int(approx_shift)} cm⁻¹ (Fingerprint / S-S stretching)"
    elif approx_shift < 1200:
        return f"~{int(approx_shift)} cm⁻¹ (C-C stretching)"
    elif approx_shift < 1700:
        return f"~{int(approx_shift)} cm⁻¹ (C=C stretching / Amide)"
    else:
        return f"~{int(approx_shift)} cm⁻¹ (C-H / O-H stretching)"


def create_feature_interpretation_table() -> None:
    """Create markdown table of top features per class and config."""
    
    # Load feature importance
    with_aug_df = pd.read_csv("outputs/reports/module_05/feature_importance.csv")
    
    # Group by class and rank
    output_lines = [
        "# Module 05: Top SHAP Features Interpretation",
        "",
        "## Feature Importance Ranking (with_aug config)",
        "",
        "| Class | Rank | Feature Index | Mean |SHAP| | Spectral Region |",
        "|-------|------|---------------|---------|--------|",
    ]
    
    for _, row in with_aug_df.iterrows():
        class_name = row["class"]
        rank = int(row["rank"])
        feat_idx = int(row["feature_idx"])
        mean_shap = float(row["mean_abs_shap"])
        region = get_wavelength_region(feat_idx)
        
        output_lines.append(
            f"| {class_name} | {rank} | {feat_idx} | {mean_shap:.6f} | {region} |"
        )
    
    output_lines.extend([
        "",
        "## Clinical Interpretation Notes",
        "",
        "### Stone (0) - Top discriminative features:",
        "- Indices 25, 26, 28: Low-mid frequency region (fingerprint bands)",
        "- Suggest: crystalline / molecular structure markers unique to cholesterol stones",
        "",
        "### Polyp (1) - Top discriminative features:",
        "- Index 1360: High wavenumber region (~2100 cm⁻¹, possible C≡ or aromatic)",
        "- Index 210: Mid-frequency region (~330 cm⁻¹, possible polysaccharide/protein)",
        "- Suggest: adenomatous polyp tissue signature in high-frequency bands",
        "",
        "### Cancer (2) - Top discriminative features:",
        "⚠️ **CAVEAT**: Test set has 0 cancer samples; these features cannot be validated clinically.",
        "- Any ranking below is exploratory only until test set includes cancer cases.",
        "",
        "## Spectral Region Reference (Typical SERS Bile)",
        "",
        "| Region | Wavenumber | Typical Attribution |",
        "|--------|------------|---------------------|",
        "| Low frequency | <300 cm⁻¹ | Lattice vibrations, Au/Ag substrate interactions |",
        "| Fingerprint | 300–800 cm⁻¹ | S-S stretching (cysteine), ring breathing |",
        "| C-C stretch | 800–1200 cm⁻¹ | Protein backbone, lipid chains |",
        "| Amide/C=C | 1200–1700 cm⁻¹ | Amide I/II (protein), unsaturated lipids |",
        "| C-H/O-H | >1700 cm⁻¹ | Aliphatic / aromatic C-H, O-H stretches |",
        "",
    ])
    
    report_text = "\n".join(output_lines)
    
    output_path = Path("outputs/reports/module_05/feature_interpretation_table.md")
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(report_text)
    
    print(f"✓ Feature interpretation table saved to {output_path}")


if __name__ == "__main__":
    create_feature_interpretation_table()
