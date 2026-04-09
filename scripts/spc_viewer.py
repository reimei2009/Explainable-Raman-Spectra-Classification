"""
spc_viewer.py - View and convert .spc files to CSV/Excel.

Usage examples
--------------
# View a single file (shows plot, prints data):
    python scripts/spc_viewer.py "Data/different potential/F35S/-200_1.spc"

# Convert single file to CSV:
    python scripts/spc_viewer.py "Data/different potential/F35S/-200_1.spc" --export csv

# Convert ALL .spc files in a sample folder to CSV:
    python scripts/spc_viewer.py "Data/different potential/F35S" --export csv

# Convert entire dataset to CSV (all samples):
    python scripts/spc_viewer.py "Data/different potential" --export csv --all

# Export to Excel instead of CSV:
    python scripts/spc_viewer.py "Data/different potential/F35S" --export xlsx
"""

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib
matplotlib.rcParams["font.family"] = "DejaVu Sans"   # fix superscript-minus glyph warning
import matplotlib.pyplot as plt
import pandas as pd

from src.data_loader import RamanDataLoader
from src.utils import setup_logging, ensure_dir

setup_logging()
loader = RamanDataLoader(verbose=True)


# ── helpers ────────────────────────────────────────────────────────────────

def view_file(spc_path: Path, export: str = None):
    """Load one .spc file, print info, optionally export and plot."""
    spectrum = loader.load_spc(spc_path)
    wn = spectrum["wavenumber"]
    inten = spectrum["intensity"]

    print(f"\n{'='*60}")
    print(f"  File     : {spc_path.name}")
    print(f"  Points   : {len(wn)}")
    print(f"  WN range : {wn.min():.2f} – {wn.max():.2f} cm⁻¹")
    print(f"  Int range: {inten.min():.4f} – {inten.max():.4f}")
    print(f"{'='*60}")

    df = pd.DataFrame({"wavenumber_cm1": wn, "intensity": inten})
    print(df.head(10).to_string(index=False))
    print(f"  ... ({len(df)} rows total)")

    if export:
        out_dir = PROJECT_ROOT / "outputs" / "csv"
        ensure_dir(out_dir)
        stem = spc_path.stem
        if export == "csv":
            out_path = out_dir / f"{stem}.csv"
            df.to_csv(out_path, index=False)
        elif export == "xlsx":
            out_path = out_dir / f"{stem}.xlsx"
            df.to_excel(out_path, index=False)
        print(f"\n✅ Exported: {out_path}")

    # Always show plot when viewing single file
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(wn, inten, lw=0.8, color="steelblue")
    ax.set_xlabel("Wavenumber (cm\u207b\u00b9)")
    ax.set_ylabel("Intensity")
    ax.set_title(f"Raman Spectrum – {spc_path.parent.name} / {spc_path.name}")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    # Save figure
    fig_dir = PROJECT_ROOT / "outputs" / "figures"
    ensure_dir(fig_dir)
    fig_path = fig_dir / f"{spc_path.stem}_view.png"
    fig.savefig(fig_path, dpi=150)
    print(f"📊 Plot saved: {fig_path}")
    plt.show()


def convert_folder(folder: Path, export: str):
    """Convert all .spc files in a sample folder."""
    spc_files = sorted(folder.glob("*.spc")) + sorted(folder.glob("*.SPC"))
    spc_files = sorted(set(spc_files))

    if not spc_files:
        print(f"No .spc files found in {folder}")
        return

    out_dir = PROJECT_ROOT / "outputs" / "csv" / folder.name
    ensure_dir(out_dir)

    dfs = []
    for fp in spc_files:
        try:
            sp = loader.load_spc(fp)
            df = pd.DataFrame({"wavenumber_cm1": sp["wavenumber"], "intensity": sp["intensity"]})
            pot, rep = loader.parse_filename(fp.name)
            df["potential_mV"] = pot
            df["replicate"] = rep
            dfs.append(df)

            if export == "csv":
                out_path = out_dir / f"{fp.stem}.csv"
                df[["wavenumber_cm1","intensity"]].to_csv(out_path, index=False)
            elif export == "xlsx":
                out_path = out_dir / f"{fp.stem}.xlsx"
                df[["wavenumber_cm1","intensity"]].to_excel(out_path, index=False)
        except Exception as e:
            print(f"  ⚠ Skipped {fp.name}: {e}")

    print(f"\n✅ Exported {len(dfs)} files → {out_dir}")

    # Also save a combined file
    if dfs:
        combined = pd.concat(dfs, ignore_index=True)
        if export == "csv":
            combined_path = out_dir / f"{folder.name}_ALL.csv"
            combined.to_csv(combined_path, index=False)
        elif export == "xlsx":
            combined_path = out_dir / f"{folder.name}_ALL.xlsx"
            combined.to_excel(combined_path, index=False)
        print(f"📋 Combined file: {combined_path}  ({len(combined)} rows)")


def convert_all_samples(data_root: Path, export: str):
    """Convert every sample folder under data_root."""
    sample_dirs = [d for d in sorted(data_root.iterdir())
                   if d.is_dir() and any(d.glob("*.spc"))]
    print(f"Found {len(sample_dirs)} sample folders.")
    for sd in sample_dirs:
        print(f"\n── {sd.name} ──")
        convert_folder(sd, export)
    print(f"\n✅ All done. Files saved under outputs/csv/")


# ── CLI ────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="View or convert Raman .spc files to CSV/Excel."
    )
    parser.add_argument("path", help="Path to .spc file, sample folder, or data root")
    parser.add_argument("--export", choices=["csv", "xlsx"], default=None,
                        help="Export format (default: just display)")
    parser.add_argument("--all", action="store_true",
                        help="Process all sample sub-folders under <path>")
    args = parser.parse_args()

    target = Path(args.path)
    if not target.exists():
        # Try relative to project root
        target = PROJECT_ROOT / args.path
    if not target.exists():
        print(f"❌ Path not found: {args.path}")
        sys.exit(1)

    if target.is_file() and target.suffix.lower() == ".spc":
        view_file(target, export=args.export)

    elif target.is_dir() and args.all:
        if args.export is None:
            args.export = "csv"
        convert_all_samples(target, args.export)

    elif target.is_dir():
        if args.export:
            convert_folder(target, args.export)
        else:
            # No export flag on a folder → show first 3 spectra as quick preview
            spc_files = sorted(target.glob("*.spc"))[:3]
            if not spc_files:
                print("No .spc files found.")
                sys.exit(1)
            for fp in spc_files:
                view_file(fp, export=None)
    else:
        print(f"❌ Not an .spc file or directory: {target}")
        sys.exit(1)


if __name__ == "__main__":
    main()
