import subprocess, sys, shutil, os

# ── 1. Install ultralytics ──────────────────────────────────
subprocess.run([sys.executable, "-m", "pip", "install", "ultralytics==8.4.127", "--force-reinstall", "-q"], check=True)

# ── 2. Find ultralytics path ────────────────────────────────
result = subprocess.run(
    [sys.executable, "-c", "import ultralytics; print(ultralytics.__file__)"],
    capture_output=True, text=True
)
#ul_path = os.path.dirname(result.stdout.strip())
ul_path = os.path.dirname(result.stdout.strip().split('\n')[-1])
modules_path = os.path.join(ul_path, "nn", "modules")

# ── 3. Copy custom.py ──────────────────────────────────────
shutil.copy("custom.py", os.path.join(modules_path, "custom.py"))
print("✓ custom.py copied")

# ── 4. Patch __init__.py ───────────────────────────────────
init_path = os.path.join(modules_path, "__init__.py")
with open(init_path, "r") as f:
    content = f.read()
if "from .custom import GSConv" not in content:
    content += "\nfrom .custom import GSConv\n"
    with open(init_path, "w") as f:
        f.write(content)
print("✓ __init__.py patched")

# ── 5. Patch tasks.py ──────────────────────────────────────
import glob

tasks_path = os.path.join(ul_path, "nn", "tasks.py")
with open(tasks_path, "r") as f:
    content = f.read()
if "from ultralytics.nn.modules.custom import GSConv" not in content:
    content = content.replace(
        "from __future__ import annotations\n",
        "from __future__ import annotations\nfrom ultralytics.nn.modules.custom import GSConv\n",
        1
    )
    content = content.replace(
        "        GhostConv,\n",
        "        GhostConv,\n        GSConv,\n",
        1
    )
    with open(tasks_path, "w") as f:
        f.write(content)

for pyc in glob.glob(os.path.join(ul_path, "nn", "__pycache__", "tasks*.pyc")):
    os.remove(pyc)
print("✓ tasks.py patched")


# ── 6. Preprocess dataset (skip D30, remap D40: 4→3) ───────
import os, shutil

input_base = '/kaggle/input/datasets/aliabdelmenam/rdd-2022/RDD_SPLIT'
output_base = '/kaggle/working/RDD_4class'

for split in ['train', 'val']:
    input_label_dir = os.path.join(input_base, split, 'labels')
    output_label_dir = os.path.join(output_base, split, 'labels')
    os.makedirs(output_label_dir, exist_ok=True)

    for fname in os.listdir(input_label_dir):
        input_path = os.path.join(input_label_dir, fname)
        output_path = os.path.join(output_label_dir, fname)
        with open(input_path, 'r') as f:
            lines = f.readlines()
        new_lines = []
        for line in lines:
            parts = line.strip().split()
            cls = int(parts[0])
            if cls == 3:
                continue        # skip D30
            if cls == 4:
                parts[0] = '3'  # remap D40: 4→3
            new_lines.append(' '.join(parts) + '\n')
        with open(output_path, 'w') as f:
            f.writelines(new_lines)

    # Symlink images
    src = os.path.join(input_base, split, 'images')
    dst = os.path.join(output_base, split, 'images')
    if not os.path.exists(dst):
        os.symlink(src, dst)

print("✓ Dataset preprocessed")

# ── 7. Create data.yaml ────────────────────────────────────
import yaml
data = {
    'path': '/kaggle/working/RDD_4class',
    'train': 'train/images',
    'val': 'val/images',
    'nc': 4,
    'names': ['D00', 'D10', 'D20', 'D40']
}
with open('/kaggle/working/RDD_4class/data.yaml', 'w') as f:
    yaml.dump(data, f)
print("✓ data.yaml created")

# ── 8. Train ───────────────────────────────────────────────
from ultralytics import YOLO

model = YOLO("yolo26_dbg.yaml")
model.train(
    data="/kaggle/working/RDD_4class/data.yaml",
    epochs=20,
    imgsz=640,
    batch=16,
    device="cuda",
    project="runs",
    name="gsconv_phase1",
)