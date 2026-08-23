import subprocess, sys, shutil, os

# ── 1. Install ultralytics ──────────────────────────────────
subprocess.run([sys.executable, "-m", "pip", "install", "ultralytics==8.4.127", "-q"], check=True)

# ── 2. Find ultralytics path ────────────────────────────────
result = subprocess.run(
    [sys.executable, "-c", "import ultralytics; print(ultralytics.__file__)"],
    capture_output=True, text=True
)
ul_path = os.path.dirname(result.stdout.strip())
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
tasks_path = os.path.join(ul_path, "nn", "tasks.py")
with open(tasks_path, "r") as f:
    content = f.read()
if "GSConv" not in content:
    content = content.replace(
        "        GhostConv,\n",
        "        GhostConv,\n        GSConv,\n"
    )
    with open(tasks_path, "w") as f:
        f.write(content)
print("✓ tasks.py patched")

# ── 6. Train ───────────────────────────────────────────────
from ultralytics import YOLO

model = YOLO("yolo26_dbg.yaml")
model.train(
    data="rdd2022.yaml",
    epochs=20,
    imgsz=640,
    batch=16,
    device=0,
    project="runs",
    name="gsconv_phase1",
)