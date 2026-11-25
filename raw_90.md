For research, I would separate this into **five diagnostic layers** before attempting NCCL communication between MIG instances on NVIDIA A100.

The objective is to establish whether your environment supports the complete chain:

`MIG visibility → CUDA peer access → NCCL compatibility → NCCL communicator → collective communication`

Don't start by setting `NCCL_P2P_DISABLE=0`. That setting cannot override a hardware, driver, or MIG isolation restriction.

# 1. GPU, driver and MIG configuration

Run these on the host:

```
# GPU and driver
nvidia-smi

# Driver version
nvidia-smi --query-gpu=name,uuid,driver_version \
  --format=csv

# MIG mode
nvidia-smi --query-gpu=mig.mode.current,mig.mode.pending \
  --format=csv

# MIG instances and UUIDs
nvidia-smi -L

# GPU instance information
nvidia-smi mig -lgi

# Compute instance information
nvidia-smi mig -lci

# Available GPU instance profiles
nvidia-smi mig -lgip

# Available compute instance profiles
nvidia-smi mig -lcip
```

**What you're checking:**

- Whether MIG is enabled.
- How many GPU instances and compute instances exist.
- Whether your target instances belong to the same physical A100.
- Which MIG profiles are being used.

# 2. GPU topology and P2P capability

```
# Full GPU topology
nvidia-smi topo -m

# P2P read connectivity
nvidia-smi topo -p2p r

# P2P write connectivity
nvidia-smi topo -p2p w

# P2P atomic connectivity
nvidia-smi topo -p2p a

# NVLink connectivity
nvidia-smi topo -p2p n

# NVLink status
nvidia-smi nvlink --status
```

For MIG, don't interpret the physical GPU topology matrix as proof that two MIG devices can communicate.

The more important test is CUDA peer accessibility between the actual visible MIG devices.

# 3. CUDA runtime and device-level diagnostics

Check the installed CUDA toolkit, runtime and driver compatibility:

```
# CUDA compiler
nvcc --version

# CUDA installation
echo $CUDA_HOME
echo $CUDA_PATH

# CUDA libraries
ldconfig -p | grep -E 'libcuda|libcudart'

# Driver API
cat /proc/driver/nvidia/version

# GPU device access
ls -l /dev/nvidia*
```

Now inspect the CUDA-visible MIG devices using Python:

```
python - <<'PY'
import torch

print("PyTorch:", torch.__version__)
print("CUDA runtime:", torch.version.cuda)
print("CUDA available:", torch.cuda.is_available())
print("Visible devices:", torch.cuda.device_count())

for i in range(torch.cuda.device_count()):
    p = torch.cuda.get_device_properties(i)
    print(f"\nDevice {i}: {p.name}")
    print("Memory:", p.total_memory / 1024**3, "GB")
    print("Compute capability:", p.major, p.minor)

print("\nPeer access:")
for i in range(torch.cuda.device_count()):
    for j in range(torch.cuda.device_count()):
        if i != j:
            try:
                print(
                    f"{i} -> {j}:",
                    torch.cuda.can_device_access_peer(i, j)
                )
            except Exception as e:
                print(f"{i} -> {j}: {e}")
PY
```

Run this with two distinct MIG UUIDs exposed:

```
CUDA_VISIBLE_DEVICES=MIG-UUID-1,MIG-UUID-2 python cuda_check.py
```

(Important) A CUDA runtime may enumerate only one MIG device with older driver generations. If you see one device instead of two, investigate driver compatibility before proceeding.

You can also use NVIDIA's CUDA samples:

```
# If CUDA samples are installed
./deviceQuery

# Peer-to-peer bandwidth and latency
./p2pBandwidthLatencyTest
```

These tests are useful for distinguishing device enumeration, peer-access capability and actual transfer behavior.

# 4. NCCL installation and version diagnostics

This is the most important layer for your research.

```
# NCCL library version
ldconfig -p | grep nccl

# Installed NCCL packages (Ubuntu/Debian)
dpkg -l | grep -i nccl

# CUDA and NCCL Python package information
python - <<'PY'
import torch

print("CUDA runtime:", torch.version.cuda)
print("NCCL version:", torch.cuda.nccl.version())

try:
    print("CUDA driver:", torch.cuda.driver_version())
except Exception:
    print("Check driver using nvidia-smi")
PY
```

Check which NCCL library your Python environment actually loads:

```
python - <<'PY'
import torch
print(torch.__file__)
print(torch.cuda.nccl.version())
PY

find /usr /opt -name 'libnccl.so*' 2>/dev/null
```

**Your target compatibility baseline:**

| Component        | Requirement                                |
| ---------------- | ------------------------------------------ |
| GPU              | NVIDIA A100                                |
| Driver           | CUDA 13-compatible driver, typically R580+ |
| NCCL             | 2.31+                                      |
| MIG              | Distinct MIG instance per rank             |
| MNNVL            | Disabled                                   |
| NCCL MIG support | Experimental                               |

NVIDIA explicitly documents these MIG requirements. NCCL 2.31+ support is experimental, so meeting the version requirements does not guarantee every topology will work.

Check the relevant environment variables:

```
env | grep -E \
'CUDA|NCCL|NVIDIA|LD_LIBRARY_PATH|PATH'
```

For the eventual MIG experiment, you will want:

```
export NCCL_MNNVL_ENABLE=0
export NCCL_DEBUG=INFO
export NCCL_DEBUG_SUBSYS=INIT,BOOTSTRAP,GRAPH,P2P,SHM,NET
```

Don't set transport overrides yet. First observe NCCL's automatic transport selection.

# 5. NCCL communication diagnostics

Once the preceding checks pass, the next step is to test NCCL independently of XGBoost.

First check whether `nccl-tests` is already installed:

```
which all_reduce_perf
which all_gather_perf
```

If not, build NVIDIA's official test suite:

```
git clone https://github.com/NVIDIA/nccl-tests.git

cd nccl-tests

make -j MPI=1 \
  CUDA_HOME=/usr/local/cuda \
  NCCL_HOME=/usr
```

Adjust `NCCL_HOME` to the actual NCCL installation prefix if necessary.

Check the resulting binaries:

```
ls -lh build/
```

Run a two-MIG AllReduce test:

```
CUDA_VISIBLE_DEVICES=MIG-UUID-1,MIG-UUID-2 \
NCCL_MNNVL_ENABLE=0 \
NCCL_DEBUG=INFO \
NCCL_DEBUG_SUBSYS=INIT,BOOTSTRAP,GRAPH,P2P,SHM,NET \
NCCL_TOPO_DUMP_FILE=/tmp/nccl-topology.xml \
./build/all_reduce_perf \
-b 8 -e 128M -f 2 -g 2 -w 5 -n 20
```

Then test AllGather:

```
CUDA_VISIBLE_DEVICES=MIG-UUID-1,MIG-UUID-2 \
NCCL_MNNVL_ENABLE=0 \
NCCL_DEBUG=INFO \
./build/all_gather_perf \
-b 8 -e 128M -f 2 -g 2 -w 5 -n 20
```

These tests establish whether a communicator can be initialized and perform actual collective operations, rather than merely checking whether CUDA can see the GPUs.

NVIDIA's official test suite: NVIDIA/nccl-tests.

**Additional NCCL 2.31+ diagnostics**

You can enable NCCL's built-in checks during communicator initialization:

```
export NCCL_RUN_DIAGNOSTICS=1
export NCCL_RUN_RAS_DIAGNOSTICS=1
```

These enable active and passive diagnostics, including GPU peer-access and system configuration checks.

# 6. Check XGBoost separately

Since your eventual target is distributed XGBoost, record these details too:

```
python - <<'PY'
import xgboost as xgb

print("XGBoost version:", xgb.__version__)
print("Build information:")
print(xgb.build_info())
PY
```

And:

```
python -m pip show xgboost
python -m pip show dask
python -m pip show distributed
```

This matters because a successful NCCL AllReduce test does not prove that XGBoost's distributed backend can use the same MIG topology.

# 7. Consolidated checklist

Use this to decide whether your environment is ready for the next stage.

| Diagnostic            | What you need to establish              |
| --------------------- | --------------------------------------- |
| `nvidia-smi`          | Driver supports CUDA 13                 |
| `nvidia-smi -L`       | Multiple MIG devices exist              |
| `nvidia-smi mig -lgi` | Correct GPU instance configuration      |
| `nvidia-smi topo -m`  | Physical topology                       |
| CUDA device query     | Both MIG devices enumerated             |
| CUDA peer access      | Actual peer-access capability           |
| NCCL version          | 2.31 or later                           |
| NCCL AllReduce        | Communicator and collective work        |
| NCCL AllGather        | Additional collective works             |
| XGBoost build info    | Installed XGBoost build characteristics |

## Where to read the authoritative documentation

1. NCCL communicator and MIG support
2. NVIDIA MIG User Guide
3. NCCL environment variables
4. NCCL performance and correctness tests

**My suggestion:** Collect the outputs of sections 1–6 before changing any configuration. Those outputs will tell us whether the problem is driver compatibility, MIG enumeration, CUDA peer access, NCCL initialization, or XGBoost integration.

One final distinction worth remembering: NCCL's experimental MIG support does not mean that all communication uses direct NVLink P2P. NCCL can establish a communicator and select a different transport. Successful AllReduce is the meaningful functional test, not simply seeing P2P marked `Yes`.
