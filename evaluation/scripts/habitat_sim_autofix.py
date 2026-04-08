import os
import shutil
import site
import glob
import importlib
from pathlib import Path

def find_replacement_file() -> Path:
    script_dir = Path(__file__).resolve().parent
    return script_dir.parent / "replace" / "common.py"


def find_target_common_py() -> Path | None:
    # Preferred: resolve from the active Python environment import path.
    try:
        module = importlib.import_module("habitat_sim.utils.common")
        module_path = Path(module.__file__).resolve()
        if module_path.exists():
            return module_path
    except Exception:
        pass

    # Fallback: scan site-packages for common install layouts.
    patterns = (
        "habitat_sim-*.egg/habitat_sim/utils/common.py",
        "habitat_sim*/habitat_sim/utils/common.py",
        "habitat_sim/utils/common.py",
    )
    for pkg_dir in site.getsitepackages():
        for pattern in patterns:
            matches = glob.glob(os.path.join(pkg_dir, pattern))
            if matches:
                return Path(matches[0]).resolve()

    # Repo fallback: useful for editable/local source checkouts.
    script_dir = Path(__file__).resolve().parent
    repo_common = (
        script_dir.parent.parent / "packages" / "habitat-sim" / "habitat_sim" / "utils" / "common.py"
    )
    if repo_common.exists():
        return repo_common.resolve()

    return None


replacement_file = find_replacement_file()
target_file = find_target_common_py()

if not replacement_file.exists():
    print(f"Error: Replacement file {replacement_file} not found!")
elif target_file is None:
    print("Error: habitat_sim/utils/common.py not found!")
else:
    shutil.copy2(replacement_file, target_file)
    print(f"Replaced {target_file} with {replacement_file}")
