"""
visualize_dataset.py - Comprehensive visualization generation for the dataset.

Run from project root:
    python scripts/visualize_dataset.py

Features:
1. Generate per-sample waterfall plots
2. Generate disease-grouped heatmaps
3. Generate age distribution plots
4. Generate gender comparison plots
5. Create summary statistics and reports
"""

import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict

from src.data_loader import RamanDataLoader
from src.utils import ensure_dir, get_project_root, setup_logging
from src.visualization import RamanVisualizer

# ---------------------------------------------------------------------------

def parse_sample_name(name):
    """Parse sample name: <Gender><Age><Disease>"""
    gender = "F" if name[0].upper() == "F" else "M"
    i = 1
    while i < len(name) and name[i].isdigit():
        i += 1
    age = int(name[1:i])
    disease = name[i:]
    return gender, age, disease

def load_all_samples(csv_root):
    """Load metadata for all samples."""
    samples = []
    for sample_dir in sorted(csv_root.iterdir()):
        if not sample_dir.is_dir():
            continue
        name = sample_dir.name
        gender, age, disease = parse_sample_name(name)
        
        # Count CSV files (approximation of spectra)
        csv_count = len(list(sample_dir.glob("*.csv")))
        
        samples.append({
            "sample": name,
            "gender": gender,
            "age": age,
            "disease": disease,
            "n_spectra": csv_count,
        })
    
    return pd.DataFrame(samples)

def generate_disease_comparison_plots(df, figures_dir):
    """Create plots comparing different diseases."""
    log = logging.getLogger(__name__)
    
    # Disease distribution pie chart
    fig, ax = plt.subplots(figsize=(8, 6))
    disease_counts = df["disease"].value_counts()
    colors = plt.cm.Set3(range(len(disease_counts)))
    ax.pie(disease_counts.values, labels=disease_counts.index, autopct="%1.1f%%",
           colors=colors, startangle=90)
    ax.set_title("Disease Distribution (Deduplicated Dataset)", fontsize=12, fontweight="bold")
    plt.tight_layout()
    path = figures_dir / "disease_distribution.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info("Saved: %s", path)
    
    # Disease vs Age scatter plot
    fig, ax = plt.subplots(figsize=(10, 6))
    diseases = df["disease"].unique()
    colors_map = {d: plt.cm.tab10(i) for i, d in enumerate(diseases)}
    
    for disease in sorted(diseases):
        subset = df[df["disease"] == disease]
        ax.scatter(subset["age"], subset["n_spectra"], label=disease, 
                  s=100, alpha=0.6, color=colors_map[disease])
    
    ax.set_xlabel("Age (years)", fontsize=10)
    ax.set_ylabel("Number of Spectra", fontsize=10)
    ax.set_title("Disease vs Age vs Spectra Count", fontsize=12, fontweight="bold")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    path = figures_dir / "disease_age_scatter.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info("Saved: %s", path)

def generate_gender_age_plots(df, figures_dir):
    """Create gender and age distribution plots."""
    log = logging.getLogger(__name__)
    
    # Gender distribution
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    
    # Gender pie chart
    gender_counts = df["gender"].value_counts()
    axes[0].pie(gender_counts.values, labels=gender_counts.index, autopct="%1.1f%%",
               colors=["#FF69B4", "#4169E1"], startangle=90)
    axes[0].set_title("Gender Distribution", fontsize=11, fontweight="bold")
    
    # Age histogram
    axes[1].hist(df["age"], bins=15, color="steelblue", edgecolor="black", alpha=0.7)
    axes[1].set_xlabel("Age (years)", fontsize=10)
    axes[1].set_ylabel("Count", fontsize=10)
    axes[1].set_title("Age Distribution", fontsize=11, fontweight="bold")
    axes[1].grid(True, alpha=0.3, axis="y")
    
    plt.tight_layout()
    path = figures_dir / "gender_age_distribution.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info("Saved: %s", path)

def generate_summary_table(df, figures_dir):
    """Create and save summary statistics table as image."""
    log = logging.getLogger(__name__)
    
    # Create summary statistics
    summary_stats = [
        ["Metric", "Value"],
        ["Total Samples", len(df)],
        ["Total Spectra", df["n_spectra"].sum()],
        ["Female / Male", f"{(df['gender']=='F').sum()} / {(df['gender']=='M').sum()}"],
        ["Age Range", f"{df['age'].min()}-{df['age'].max()} years"],
        ["Age Mean ± Std", f"{df['age'].mean():.1f} ± {df['age'].std():.1f}"],
        ["Diseases", ", ".join(sorted(df["disease"].unique()))],
    ]
    
    # Disease breakdown
    disease_breakdown = []
    for disease in sorted(df["disease"].unique()):
        count = (df["disease"] == disease).sum()
        pct = 100 * count / len(df)
        disease_breakdown.append([f"{disease}", f"{count} ({pct:.1f}%)"])
    
    print("\n" + "="*60)
    print("DATASET SUMMARY")
    print("="*60)
    for row in summary_stats:
        print(f"{row[0]:20s} {row[1]}")
    
    print("\nDISEASE BREAKDOWN:")
    for disease, count in disease_breakdown:
        print(f"  {disease:10s}: {count}")
    print("="*60 + "\n")

def main():
    setup_logging()
    log = logging.getLogger(__name__)
    
    root = get_project_root()
    csv_root = root / "Data" / "different potential" / "csv"
    figures_dir = root / "outputs" / "figures"
    ensure_dir(figures_dir)
    
    if not csv_root.exists():
        log.error("CSV data not found: %s", csv_root)
        sys.exit(1)
    
    log.info("Loading dataset metadata...")
    df = load_all_samples(csv_root)
    
    print(f"\n✓ Loaded {len(df)} samples")
    print(f"  - Total spectra: {df['n_spectra'].sum()}")
    print(f"  - Diseases: {', '.join(sorted(df['disease'].unique()))}\n")
    
    # Generate plots
    print("Generating visualizations...\n")
    
    generate_disease_comparison_plots(df, figures_dir)
    generate_gender_age_plots(df, figures_dir)
    generate_summary_table(df, figures_dir)
    
    # Generate per-sample waterfall plots
    print("\nGenerating per-sample waterfall plots...")
    loader = RamanDataLoader(verbose=False)
    viz = RamanVisualizer(output_dir=figures_dir / "per_sample")
    
    per_sample_dir = figures_dir / "per_sample"
    ensure_dir(per_sample_dir)
    
    success = 0
    for idx, (_, row) in enumerate(df.iterrows(), 1):
        sample_name = row["sample"]
        sample_dir = csv_root / sample_name
        
        figure_path = per_sample_dir / f"{sample_name}_waterfall.png"
        if figure_path.exists():
            continue
        
        try:
            sample_data = loader.load_sample_folder(str(sample_dir))
            if sample_data:
                fig = viz.plot_potential_series(sample_data, sample_name=sample_name)
                viz.save_figure(fig, f"{sample_name}_waterfall.png")
                plt.close(fig)
                success += 1
                if idx % 5 == 0:
                    print(f"  [{idx}/{len(df)}] Generated {success} waterfall plots")
        except Exception as e:
            log.warning("Could not generate plot for %s: %s", sample_name, e)
    
    print(f"\n✓ Generated {success} waterfall plots")
    print(f"✓ All figures saved to: {figures_dir}\n")


if __name__ == "__main__":
    main()
