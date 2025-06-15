# ActiveCircuitDiscovery - L40S GPU Deployment Guide

## Table of Contents
- [Overview](#overview)
- [Golden Image Snapshot](#golden-image-snapshot)
- [Deployment Options](#deployment-options)
- [Quick Start](#quick-start)
- [Troubleshooting](#troubleshooting)
- [Cost Management](#cost-management)

## Overview

This guide covers deployment of the ActiveCircuitDiscovery framework on DigitalOcean L40S GPU droplets. The framework performs transformer circuit analysis using Active Inference and mechanistic interpretability.

### L40S GPU Specifications
- **Memory:** 46GB GDDR6 (perfect for large transformer models)
- **Architecture:** Ada Lovelace with 4th-gen Tensor Cores
- **CUDA:** 12.1 compatibility (matches DigitalOcean AI/ML image)
- **Cost:** ~$0.76/hour ($1.52 per 2-hour session)

## Golden Image Snapshot

### What is Included
The golden image snapshot contains a fully configured environment:

✅ **Base System:**
- Ubuntu 22.04.4 LTS
- NVIDIA drivers 535.216.01
- CUDA 12.1 toolkit
- nvidia-container-toolkit 1.71.0

✅ **Python Environment:**
- Python 3.10.12 with virtual environment
- PyTorch 2.5.1 with CUDA 12.1 support
- All dependencies pre-installed and tested

✅ **Key Libraries (Tested Versions):**
```
torch==2.5.1 (with CUDA 12.1)
transformers==4.52.4
transformer-lens==1.17.0
pymdp==0.0.1 (Active Inference)
numpy==2.1.2
matplotlib==3.10.3
```

✅ **Project Setup:**
- ActiveCircuitDiscovery repository cloned
- Virtual environment activated by default
- Environment variables configured
- Helpful aliases pre-configured

### Creating the Golden Image Snapshot

**Via DigitalOcean Control Panel:**
1. Navigate to your L40S Droplet (IP: 178.128.230.171)
2. Go to "Snapshots" tab
3. Click "Take Snapshot"
4. **Name:** `ActiveCircuitDiscovery-L40S-Golden-20250615`
5. **Description:** "Production-ready L40S environment with all dependencies"
6. Wait 5-15 minutes for completion

## Deployment Options

### Option 1: Golden Image Deployment (Recommended)

**Advantages:**
- ⚡ **Instant Setup:** Ready in 2-3 minutes
- 💰 **Cost Effective:** No setup time charges
- 🔒 **Reliable:** Pre-tested configuration
- 🎯 **Consistent:** Identical environment every time

**Steps:**
1. **Create Droplet from Snapshot:**
   - DigitalOcean Control Panel → "Create" → "Droplets"
   - **Image:** "Snapshots" → Select golden image
   - **Size:** L40S GPU droplet
   - **Region:** Any region with L40S availability
   - **SSH Keys:** Add your public key
   - Click "Create Droplet"

2. **Immediate Usage:**
   ```bash
   ssh ubuntu@<NEW_DROPLET_IP>
   cd ActiveCircuitDiscovery
   # Virtual environment auto-activated
   python run_complete_experiment.py  # Ready to run\!
   ```

## Quick Start

### From Golden Image (2 minutes)
```bash
# 1. Create droplet from snapshot via DigitalOcean control panel
# 2. SSH to new droplet
ssh ubuntu@<DROPLET_IP>

# 3. Verify environment
python test_environment.py

# 4. Run experiment
cd ActiveCircuitDiscovery
python run_complete_experiment.py

# 5. Monitor progress
tail -f logs/experiment.log
```

## Environment Details

### Pre-configured Aliases
```bash
alias experiment="cd ~/ActiveCircuitDiscovery && python run_complete_experiment.py"
alias gpu="nvidia-smi"
alias gpumon="~/gpu_monitor.sh"
alias logs="tail -f ~/ActiveCircuitDiscovery/logs/startup.log"
alias activate="source ~/ActiveCircuitDiscovery/venv/bin/activate"
```

### Directory Structure
```
/home/ubuntu/ActiveCircuitDiscovery/
├── src/                          # Framework source code
├── results/                      # Experiment outputs
├── logs/                        # Runtime logs
├── venv/                        # Python virtual environment
├── run_complete_experiment.py   # Main experiment script
├── requirements.txt             # Tested dependencies
└── L40S_GPU_DEPLOYMENT_GUIDE.md # This guide
```

## Cost Management

### Pricing Structure
- **L40S GPU:** $0.76/hour
- **Typical Experiment:** 30-60 minutes
- **Average Cost per Run:** $0.38 - $0.76

### Cost Optimization Strategies

**1. Golden Image Benefits:**
- Save 10-15 minutes setup time per session
- Immediate experiment execution
- Cost savings: ~$0.13-$0.19 per deployment

**2. Efficient Workflows:**
```bash
# Quick experiment cycle
ssh ubuntu@<DROPLET_IP>
cd ActiveCircuitDiscovery && python run_complete_experiment.py
# Wait for completion, then destroy droplet
```

## Support and Documentation

### Key Files
- `requirements.txt` - Tested dependency versions
- `startup-script.yaml` - Fresh deployment automation
- `startup-script-golden.yaml` - Golden image deployment
- `test_environment.py` - Environment verification

### GitHub Integration
- Repository: https://github.com/SharathSPhD/ActiveCIrcuitDiscovery
- Authentication: Token-based (configured)
- Branches: Experiment branches for results

---

**Generated:** 2025-06-15
**Environment:** L40S GPU with 46GB memory
**Status:** Production Ready ✅
**Last Tested:** 2025-06-15

For questions or issues, refer to project documentation or create GitHub issues.
