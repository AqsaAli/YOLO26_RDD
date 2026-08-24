import subprocess, sys, shutil, os, glob

# ── 1. Install ultralytics ──────────────────────────────────
subprocess.run([sys.executable, "-m", "pip", "install", "ultralytics==8.4.127", "--force-reinstall", "-q"], check=True)

# ── 2. Find ultralytics path ────────────────────────────────
result = subprocess.run(
    [sys.executable, "-c", "import ultralytics; print(ultralytics.__file__)"],
    capture_output=True, text=True
)
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