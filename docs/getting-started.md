# Getting Started with MONO3D

This guide provides comprehensive instructions for setting up the MONO3D development environment, from initial repository access through various deployment options including Docker containers and VSCode dev-containers.

## 📋 Prerequisites

### System Requirements

**Software Requirements:**
- **Python**: 3.10.9 (required for compatibility with PyTorch and dependencies)
- **UV**: Modern Python package manager (recommended over pip)
- **Git**: Version control system
- **Docker**: For containerized deployment (optional)
- **VSCode**: For dev-container support (optional)

**Hardware Requirements:**

**Minimum Configuration (CPU-only):**
- 8 GB RAM
- 50 GB available storage
- Multi-core CPU (4+ cores recommended)

**Recommended Configuration (GPU-accelerated):**
- 16+ GB RAM
- 100+ GB available storage  
- NVIDIA GPU with 8+ GB VRAM (RTX 3080/4080 or equivalent)
- CUDA 11.8+ compatible drivers

**Optimal Configuration (Research/Training):**
- 32+ GB RAM
- 200+ GB NVMe storage
- NVIDIA GPU with 16+ GB VRAM (RTX 4090, A6000, or equivalent)
- CUDA 12.0+ compatible drivers

---

## 🔐 Repository Access

### 1. Clone the Repository

```bash
# Clone the repository
git clone https://github.com/alberto-rota/MONO3D.git
cd MONO3D
```

### 2. Verify Repository Structure

```bash
# Check that all key directories are present
ls -la
# Expected output should include:
# - pipelines/          # Core pipeline implementations
# - docs/               # Documentation
# - docker-compose.yaml # Container orchestration
# - .devcontainer/      # VSCode dev-container configuration
# - requirements.txt    # Python dependencies
```

---

## 🐍 Python Environment Setup

### 1. Install UV Package Manager

UV is a fast Python package manager that provides better dependency resolution and caching than pip:

```bash
# Install UV (on macOS/Linux)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Install UV (on Windows)
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"

# Verify installation
uv --version
```

### 2. Create and Activate Virtual Environment

```bash
# Create virtual environment with Python 3.10.9
uv venv --python 3.10.9

# Activate virtual environment
# On Linux/macOS:
source .venv/bin/activate

# On Windows:
.venv\Scripts\activate
```

### 3. Install Dependencies

```bash
# Install all required packages
uv pip install -r requirements.txt

# Verify installation
python -c "import torch; print(f'PyTorch version: {torch.__version__}')"
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
```

### 4. Environment Variables

```bash
# Set Python path for module imports
export PYTHONPATH="${PYTHONPATH}:$(pwd)"

# Set dataset root directory (modify path as needed)
export DATASET_ROOTDIR="/path/to/your/datasets"
```

---

## 🐳 Docker Deployment

The project includes a comprehensive Docker setup with GPU support and development tools.

### 1. Docker Compose Configuration

The `docker-compose.yaml` file provides two services:

- **mono3dev**: Development container with GPU support
- **mono3docs**: Documentation server

### 2. Hardware-Specific Setup

**For GPU-enabled systems:**

The default configuration includes GPU support. Ensure you have:

```bash
# Check NVIDIA Docker runtime
nvidia-docker --version

# Verify GPU access
nvidia-smi
```

**For CPU-only systems:**

Modify `docker-compose.yaml` to disable GPU features:

```yaml
services:
  mono3dev:
    # Comment out GPU-specific configurations
    # runtime: nvidia
    environment:
      # - NVIDIA_VISIBLE_DEVICES=all
      # - NVIDIA_DRIVER_CAPABILITIES=compute,utility
      - PYTHONPATH=/workspace
      - DATASET_ROOTDIR=/DATA
    # Comment out GPU deployment section
    # deploy:
    #   resources:
    #     reservations:
    #       devices:
    #         - capabilities: [gpu]
```

### 3. Dataset Mounting

Configure dataset paths in `docker-compose.yaml`:

```yaml
volumes:
  # Mount your dataset directory
  # Format: /host/path:/container/path
  - /home/alberto/SCARED:/DATA/SCARED
  
  # Add additional datasets as needed
  # - /host/path/to/CHOLEC80:/DATA/CHOLEC80
  # - /host/path/to/EndoMapper:/DATA/EndoMapper
```

### 4. Container Operations

**Start the development environment:**

```bash
# Start all services
docker-compose up -d

# Start only development container
docker-compose up -d mono3dev

# Start only documentation server
docker-compose up -d mono3docs
```

**Access the development container:**

```bash
# Get a shell in the container
docker-compose exec mono3dev bash

# The UV virtual environment needs to be activated inside the container
source /workspace/.venv/bin/activate

# Verify environment
python -c "import torch; print(f'PyTorch version: {torch.__version__}')"
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
```

**Access the documentation:**

The documentation is served at `http://localhost:8000` when the `mono3docs` service is running.

```bash
# View documentation
curl http://localhost:8000
# Or open in browser: http://localhost:8000
```

**Container management:**

```bash
# View running containers
docker-compose ps

# View logs
docker-compose logs mono3dev

# Stop services
docker-compose down

# Rebuild containers (after changes)
docker-compose build --no-cache
```

---

## 💻 VSCode Dev-Container Setup

The project includes a preconfigured development container for VSCode with all necessary extensions and settings.

### 1. Prerequisites

```bash
# Install VSCode extensions
code --install-extension ms-vscode-remote.remote-containers
```

### 2. Dev-Container Configuration

The `.devcontainer/devcontainer.json` file includes:

- **GPU Support**: Automatic GPU passthrough with NVIDIA runtime
- **Extensions**: Python, Pylance, Docker, Jupyter, Git tools
- **Settings**: Python interpreter, linting, formatting (Black)
- **Mounts**: Workspace, virtual environment, results, and datasets

### 3. Opening the Project

**Method 1: Command Palette**
1. Open VSCode
2. Press `Ctrl+Shift+P` (or `Cmd+Shift+P` on macOS)
3. Type "Remote-Containers: Open Folder in Container"
4. Select the MONO3D folder

**Method 2: Notification**
1. Open the MONO3D folder in VSCode
2. Click "Reopen in Container" when prompted

**Method 3: Command Line**
```bash
# From the MONO3D directory
code .
# Then use Command Palette as in Method 1
```

### 4. Container Customization

**Modify container settings in `.devcontainer/devcontainer.json`:**

```json
{
  "customizations": {
    "vscode": {
      "settings": {
        "python.defaultInterpreterPath": "/workspace/.venv/bin/python",
        "python.linting.enabled": true,
        "python.linting.pylintEnabled": true,
        "python.formatting.provider": "black",
        "editor.formatOnSave": true,
        "editor.rulers": [88]
      }
    }
  }
}
```

**Add additional VSCode extensions:**

```json
{
  "customizations": {
    "vscode": {
      "extensions": [
        "ms-python.python",
        "ms-python.vscode-pylance",
        "ms-toolsai.jupyter",
        "your-additional-extension"
      ]
    }
  }
}
```

### 5. Development Workflow

**Inside the dev-container:**

```bash
# Virtual environment is automatically activated
# Verify Python environment
python --version  # Should show 3.10.9
which python      # Should show /workspace/.venv/bin/python

# Run tests
python -m pytest tests/

# Start Jupyter notebook
jupyter notebook --ip=0.0.0.0 --port=8888 --no-browser --allow-root

# Access notebooks at: http://localhost:8888
```

---

## 🚀 Quick Start Verification

### 1. Test Pipeline Imports

```python
# Test core pipeline imports
python -c "
from pipelines.features import FeatureExtractor
from pipelines.matching import MatchingPipeline  
from pipelines.depth import DepthPipeline
from pipelines.odometry import OdometryPipeline
print('✅ All pipelines imported successfully')
"
```

### 2. Test GPU Acceleration

```python
# Test GPU availability and tensor operations
python -c "
import torch
print(f'PyTorch version: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'CUDA device: {torch.cuda.get_device_name(0)}')
    print(f'CUDA memory: {torch.cuda.get_device_properties(0).total_memory // 1024**3} GB')
    # Test tensor operations
    x = torch.randn(1000, 1000).cuda()
    y = torch.matmul(x, x.T)
    print('✅ GPU tensor operations working')
else:
    print('⚠️  CUDA not available - running on CPU')
"
```

### 3. Test Dataset Access

```python
# Test dataset loading (modify paths as needed)
python -c "
import os
dataset_root = os.environ.get('DATASET_ROOTDIR', '/DATA')
print(f'Dataset root: {dataset_root}')
if os.path.exists(dataset_root):
    print(f'Available datasets: {os.listdir(dataset_root)}')
    print('✅ Dataset access working')
else:
    print('⚠️  Dataset directory not found - check DATASET_ROOTDIR')
"
```

---

## 🔧 Troubleshooting

### Common Issues

**1. Python Version Mismatch**
```bash
# Verify Python version
python --version
# Should show 3.10.9

# If incorrect, recreate virtual environment
rm -rf .venv
uv venv --python 3.10.9
source .venv/bin/activate
uv pip install -r requirements.txt
```

**2. GPU Not Detected**
```bash
# Check NVIDIA drivers
nvidia-smi

# Check Docker GPU support
docker run --rm --gpus all nvidia/cuda:11.8-base-ubuntu22.04 nvidia-smi

# Verify PyTorch GPU support
python -c "import torch; print(torch.cuda.is_available())"
```

**3. Container Build Issues**
```bash
# Clean rebuild
docker-compose down
docker system prune -a
docker-compose build --no-cache
```

**4. Permission Issues**
```bash
# Fix file permissions
sudo chown -R $USER:$USER .
chmod -R 755 .
```

### Performance Optimization

**1. GPU Memory Management**
```python
# Clear GPU cache
import torch
torch.cuda.empty_cache()

# Set memory fraction
torch.cuda.set_per_process_memory_fraction(0.8)
```

**2. Docker Performance**
```bash
# Allocate more memory to Docker
# In Docker Desktop: Settings > Resources > Memory > 8+ GB
```

---

## 📚 Next Steps

1. **[Configuration Guide](configuration.md)** - Learn about model and training parameters
2. **[Training Tutorial](training.md)** - Start training your first model
3. **[API Reference](api/)** - Explore the complete code documentation
4. **[Dataset Preparation](datasets.md)** - Prepare your surgical video data

For additional support, check the [GitHub Issues](https://github.com/alberto-rota/MONO3D/issues) or contact the development team. 