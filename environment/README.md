# Setup Environment 

```bash
mamba env create -f environment/linux.yml
mamba activate spanetv2

pip install uv 
UV_HTTP_TIMEOUT=300 uv pip install -r requirements.txt
```