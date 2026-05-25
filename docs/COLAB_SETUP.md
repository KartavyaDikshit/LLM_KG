# Section 1: Environment Setup
Run this cell first to clone your fork, install dependencies, and start the local LLM engine.

```python
import os, shutil, subprocess, time, sys

# 1. Setup Folders and cleanup
PROJECT_DIR = '/content/LLM_KG'
if os.path.exists(PROJECT_DIR): shutil.rmtree(PROJECT_DIR)

# 2. Clone YOUR Fork & Install dependencies
# Replace with your fork link if different
!git clone https://github.com/KartavyaDikshit/LLM_KG.git
%cd {PROJECT_DIR}

!pip install -r requirements.txt --quiet
!pip install langchain-community langchain-ollama pyvis tabulate PyYAML --quiet
sys.path.append(PROJECT_DIR)

# 3. Setup Ollama (Local Engine)
!sudo apt-get update && sudo apt-get install -y zstd
!curl -fsSL https://ollama.com/install.sh | sh
subprocess.Popen(["ollama", "serve"])
time.sleep(15)

# 4. Pull ONLY the most stable models
models = ["llama3", "mistral", "gemma2"]
for m in models:
    print(f"📥 Pulling {m}...")
    !ollama pull {m}

print("✅ Environment Ready!")
```
