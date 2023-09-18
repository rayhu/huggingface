# About this repo

This repo contains code and docs for running models offline.
Especially llama 2 is a powerful model to run.

## Installation

### GPU related software

Need a GPU, because many packages were using FP16, which is not natively supported on modern CPUs.

First, install the nvidia graphics driver for your card.
Make sure you have the latest version.

This display the graphics card infomation.

Then install the latest CUDA toolkit for your card. It comes with useful tools and samples. If the CUDA toolkit version is lower in your environment, the actual GPU driver on your system is what allows CUDA to run. This driver is backward compatible with older CUDA toolkit versions. However, the reverse isn't true; older drivers wouldn't support newer CUDA toolkit versions. Check current CUDA toolkit version by:

```powershell
nvidia-smi
```

After you have CUDA toolkit installed, verify your nvcc version is the same.

```powershell
nvcc --version
```

### Automatically Install everything using conda lock

The packages and dependencies are included in a conda lock file for reproducibility.

```powershell
conda create --name huggingface --file conda-explicit.txt
```

There are two files can be found in the root of the repo to restore the env. You can restore using any of the files.

* conda-explicit.txt contains all the urls and some hashes. It is great for most circumstances.
* environment.yml is more human readable. Make changes to this file if you want to change the versions of the packages.

If for some reason, you need to try different versions, follow the steps below and make changes. Otherwise, skip to the step of running the code.

### Manually Install everything

#### Install AI Frameworks

Typically, when you install PyTorch or TensorFlow, they come with their own version of CUDA toolkit. If you have CUDA 12.2 on your system but the conda channel offers PyTorch with CUDA 11.8, it's generally okay. PyTorch will use its bundled CUDA version. 11.8 is the latest version at the time of writing.

Need to have conda first, then create a new conda environment with python 3.9.

```powershell
conda create -n huggingface python=3.9
conda activate huggingface

```

Then you shall go to [pytorch website](https://pytorch.org/get-started/locally/) to generate the command to install the right pytorch. 
Select
* PyTorch version: 2.0.1
* OS: Windows
* Package: conda
* Language: Python
* Compute Platform: CUDA 11.8

Copy the generated command and run it:

```powershell
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
```

After the installation, test it out using a small python code.

```powershell
python -c "import torch; print('CUDA Enabled: ' + str(torch.cuda.is_available()));print('CUDA version: ' + torch.version.cuda)"
```

It shall display:

```powershell
CUDA Enabled: True
CUDA version: 11.8
```

It shall display the cuda version used by the pytorch library.

#### Install Hugging Face

Hugging Face makes it easy to run models by wrapping them into same pipelines.

To install transformer from hugging faces

```powershell
conda install transformers
```

Test the installation:

```powershell
python -c "from transformers import pipeline; print(pipeline('sentiment-analysis')('I love you'))"
```

It shall return a positive number close to 1.

#### Install Jupyter Lab

Jupyter Lab or Jupyter Notebook are convenient way to run the python code or make changes.

To run the code in jupyter, install it using

```powershell
conda install jupyterlab
conda install -c conda-forge ipywidgets
conda install -c conda-forge tokenizers=0.13.3
conda install -c conda-forge accelerate
```

You can check your login token to huggingface using: 

```
huggingface-cli whoami
```

If this command tells you are not logged in, you can run the command below and save a token.

```powershell
huggingface-cli login
```


Set the code to work offline `TRANSFORMERS_OFFLINE=1`

The ipywidgets is a better UI to show progress.

The tokenizers shall be at least 0.13.3 to work with transformers.

Using `low_cpu_mem_usage=True` or a `device_map` requires Accelerate

#### Export the package files

```powershell
conda list --explicit > conda-explicit.txt
conda env export > environment.yml
```


## Running the code

The scripts are tested on powershell.

```powershell
cd C:\play\huggingface
jupyter lab
```

Open the lab notes from the link, and run the code.


## Tips

### Don't mix pip and conda

There might be some troubles, when needed, start from a clean conda environment. Don't install packages to conda base environment.

### Conda channels

When possible, use the official channels, if a package can be found in conda channel, don't install from conda-forge.

The preference sequence of channels:
* conda: default channel
* nvidia: official channel for nvidia
* conda-forge: community maintained

See the differences between channels

```powershell
conda search cudatoolkit -c nvidia
conda search cudatoolkit -c conda-forge
```

### Install the latest nvidia graphics card driver

Even if the CUDA toolkit version is lower in your environment, the actual GPU driver on your system is what allows CUDA to run. This driver is backward compatible with older CUDA toolkit versions. However, the reverse isn't true; older drivers wouldn't support newer CUDA toolkit versions.

[Read Nvidia explanation of driver and CUDA](https://docs.nvidia.com/deploy/cuda-compatibility/index.html#binary-compatibility__table-toolkit-driver)

### Use a GPU

Running FP16 on CPU usually has to expand to FP32, which is a lot of trouble. Need to play with the Tensors, usually they are encapsulated, and playing with it has no performance benefits.

Such as:
```python
tensor = tensor.to(torch.float32)
```

### CUDA
A generation of NVIDIA GPU has a CUDA compute capability. For example, Kepler architecture has CC 3.0.

[The GPU CC/Generation matrix on wikipedia](https://en.wikipedia.org/wiki/CUDA)

A CUDA toolkit version supports a range of compute capabilities. For example, Toolkit 10.2  support compute capability 3.0 to 7.5.

CUDA Toolkit 11.0 dropped support of Compute capability 3.0, so for a kepler card, CUDA Toolkit versions 10.2 is the latest version still works.

[The CUDA CC/Toolkit matrix on stackoverflow](https://stackoverflow.com/questions/28932864/which-compute-capability-is-supported-by-which-cuda-versions)


So when you install pytorch for kepler car, you shall install pytorch with cuda 10.2 to leverage your GPU, which is versiuon 1.12.1. Search for `CUDA 10.2` in the pytorch old versions webpage:

https://pytorch.org/get-started/previous-versions/


### Upgrading


```powershell
conda install transformers --upgrade
```