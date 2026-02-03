"""
Diagnostic script to check NaVILA pose export prerequisites.
Run this in your navila-eval environment.
"""

import sys
from pathlib import Path

def check_python_version():
    print(f"Python version: {sys.version}")
    return sys.version_info >= (3, 10)

def check_imports():
    """Check if critical packages are installed"""
    results = {}

    packages = [
        "habitat",
        "habitat_sim",
        "torch",
        "numpy",
        "cv2",
        "PIL",
        "tqdm",
        "jsonlines",
    ]

    for pkg in packages:
        try:
            if pkg == "cv2":
                import cv2
                results[pkg] = cv2.__version__
            elif pkg == "PIL":
                from PIL import Image
                results[pkg] = Image.__version__
            else:
                mod = __import__(pkg)
                results[pkg] = getattr(mod, "__version__", "installed")
        except ImportError:
            results[pkg] = "NOT INSTALLED"

    return results

def check_directories():
    """Check if key directories exist"""
    base = Path.cwd().parent.parent  # assume we're in evaluation/scripts

    dirs_to_check = {
        "NaVILA root": base,
        "evaluation/data": base / "evaluation" / "data",
        "evaluation/data/datasets": base / "evaluation" / "data" / "datasets",
        "evaluation/data/scene_datasets": base / "evaluation" / "data" / "scene_datasets",
        "evaluation/data/datasets/R2R_VLNCE_v1-3_preprocessed": base / "evaluation" / "data" / "datasets" / "R2R_VLNCE_v1-3_preprocessed",
        "evaluation/data/datasets/RxR_VLNCE_v0": base / "evaluation" / "data" / "datasets" / "RxR_VLNCE_v0",
        "evaluation/data/scene_datasets/mp3d": base / "evaluation" / "data" / "scene_datasets" / "mp3d",
    }

    results = {}
    for name, path in dirs_to_check.items():
        if path.exists() and path.is_dir():
            try:
                count = len(list(path.iterdir()))
                results[name] = f"EXISTS ({count} items)"
            except Exception:
                results[name] = "EXISTS (can't list)"
        else:
            results[name] = "NOT FOUND"

    return results

def check_navila_dataset():
    """Check if NaVILA training dataset exists (with ScanQA layout support)"""

    navila_data_paths = [
        Path("/home/rithvik/NaVILA-Dataset"),
    ]

    results = {}

    for base_path in navila_data_paths:
        if not base_path.exists():
            results[str(base_path)] = "NOT FOUND"
            continue

        results[str(base_path)] = "FOUND"

        for subdir in ["R2R", "RxR", "Human", "EnvDrop", "ScanQA"]:
            subpath = base_path / subdir

            if not subpath.exists():
                results[f"  {subdir}/"] = "NOT FOUND"
                continue

            # Special handling for ScanQA
            if subdir == "ScanQA":
                direct_file = subpath / "annotations.json"
                ann_dir = subpath / "annotations"

                if direct_file.exists():
                    results[f"  ScanQA/annotations.json"] = "EXISTS"
                elif ann_dir.exists() and any(ann_dir.glob("*.json")):
                    count = len(list(ann_dir.glob("*.json")))
                    results[f"  ScanQA/annotations/*.json"] = f"EXISTS ({count} files)"
                else:
                    results[f"  ScanQA/annotations"] = "MISSING"

                continue

            # Standard datasets
            anno_file = subpath / "annotations.json"
            if anno_file.exists():
                results[f"  {subdir}/annotations.json"] = "EXISTS"
            else:
                results[f"  {subdir}/annotations.json"] = "MISSING"

    return results

def check_habitat_config():
    """Check if VLN-CE config files exist"""
    base = Path.cwd().parent  # evaluation/

    configs = [
        base / "habitat_extensions" / "config" / "vlnce_task.yaml",
        base / "vlnce_baselines" / "config" / "r2r_baselines" / "navila.yaml",
    ]

    results = {}
    for cfg in configs:
        results[str(cfg.relative_to(base.parent))] = "EXISTS" if cfg.exists() else "NOT FOUND"

    return results

def main():
    print("=" * 80)
    print("NaVILA Pose Export Setup Diagnostic")
    print("=" * 80)
    print()

    print("1. Python Version Check:")
    py_ok = check_python_version()
    print(f"   {'✓' if py_ok else '✗'} Python 3.10+ required")
    print()

    print("2. Package Installations:")
    packages = check_imports()
    for pkg, version in packages.items():
        status = "✓" if version != "NOT INSTALLED" else "✗"
        print(f"   {status} {pkg:20s} {version}")
    print()

    print("3. Evaluation Data Directories:")
    dirs = check_directories()
    for name, status in dirs.items():
        marker = "✓" if "EXISTS" in status else "✗"
        print(f"   {marker} {name}: {status}")
    print()

    print("4. NaVILA Training Dataset:")
    navila_data = check_navila_dataset()
    for path, status in navila_data.items():
        marker = "✓" if "FOUND" in status or "EXISTS" in status else "✗"
        print(f"   {marker} {path}: {status}")
    print()

    print("5. VLN-CE Config Files:")
    configs = check_habitat_config()
    for cfg, status in configs.items():
        print(f"   {'✓' if status == 'EXISTS' else '✗'} {cfg}: {status}")
    print()

    print("=" * 80)
    print("Summary:")
    print("=" * 80)

    critical_missing = []

    if packages.get("habitat") == "NOT INSTALLED":
        critical_missing.append("habitat-lab (v0.1.7)")
    if packages.get("habitat_sim") == "NOT INSTALLED":
        critical_missing.append("habitat-sim (v0.1.7)")

    if "NOT FOUND" in dirs.get("evaluation/data/datasets/R2R_VLNCE_v1-3_preprocessed", ""):
        critical_missing.append("R2R_VLNCE dataset")
    if "NOT FOUND" in dirs.get("evaluation/data/scene_datasets/mp3d", ""):
        critical_missing.append("MP3D scene data")

    navila_found = any(v == "FOUND" for v in navila_data.values())
    if not navila_found:
        critical_missing.append("NaVILA training dataset")

    if critical_missing:
        print("✗ CRITICAL ITEMS MISSING:")
        for item in critical_missing:
            print(f"    - {item}")
        print()
        print("See instructions below to install missing components.")
    else:
        print("✓ All critical components found!")
        print("You can proceed to pose export.")

    print()

if __name__ == "__main__":
    main()
