"""Print the library versions and GPU of the current environment as JSON (step 1, for the step-2 runs).

    python tool_env.py > results/step2/env_<conda env>.json
"""
import importlib
import json
import platform
import subprocess
import sys

out = dict(python=sys.version.split()[0], python_executable=sys.executable, platform=platform.platform())
for m in ('torch', 'torchvision', 'mmcv', 'mmengine', 'mmdet', 'mmseg', 'timm', 'numpy', 'fvcore'):
    try:
        out[m] = importlib.import_module(m).__version__
    except Exception as e:  # not installed in this environment
        out[m] = f'- ({type(e).__name__})'
try:
    import torch
    out['cuda_runtime'] = torch.version.cuda
    out['cuda_available'] = torch.cuda.is_available()
except Exception:
    pass
out['gpu'] = subprocess.run('nvidia-smi --query-gpu=name,driver_version --format=csv,noheader -i 0', shell=True,
                            capture_output=True, text=True).stdout.strip()
print(json.dumps(out, indent=1))
