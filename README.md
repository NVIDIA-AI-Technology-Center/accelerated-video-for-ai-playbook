# Accelerated Video Processing Playbook

## Introduction

In this repo you will find video data loading libraries tests, codes and useful information for accelerated video data loading for AI/ML trainings.

## License

The Video Playbook is released under the [MIT License](LICENSE).

## Installation

We recommend to start from the NVIDIA NGC PyTorch [container](https://catalog.ngc.nvidia.com/orgs/nvidia/containers/pytorch) and then install other dependencies.
To pull a PyTorch Docker from NVIDIA NGC you can copy paste the following command to your terminal
```bash
docker pull nvcr.io/nvidia/pytorch:25.03-py3
```
Keep in mind you can substitute 25.03 with the more recent release, the format is YY.MM (year.month).

Alternatively you can install your packages manually, but this is not mantained and not tested.

Depending on the libraries that you want to test in the [Video Libraries Tests.ipynb](01%20Video%20Libraries%20Tests.ipynb) and assuming we start from a PyTorch Docker, for the installation please follow the corresponding steps below.

- **DALI** [DALI](https://github.com/NVIDIA/DALI) is already installed in NVIDIA PyTorch dockers, should you need to install it again please follow the DALI docs.
- **PyNvVideoCodec** [PyNvVideoCodec](https://developer.nvidia.com/pynvvideocodec) or in short PyNvC is a simple pip install
```bash
pip install pynvvideocodec
```
If you want to access also additional resources from this package, like more coding examples, you can download a zip file from NVIDIA NGC [here](https://catalog.ngc.nvidia.com/orgs/nvidia/resources/pynvvideocodec). In both cases it is a pip install.
- **PyAv** Used by PyTorch native video dataloaders is a simple pip install.
```bash
pip install av
```
- **decord** Is the most used video data loading library among researchers recently. If you want to install the CPU version it is a simple pip install
```bash
pip install decord
```
For the GPU accelerated version it requires to clone the Github project and to follow the compilation steps provided [here](https://github.com/dmlc/decord?tab=readme-ov-file#linux).
- **openCV** It comes preinstalled in the PyTorch NVIDIA docker image for CPU. However,
In some cases we found that the native opencv installation on the docker might be broken or have some issues, there are two ways we found to work around these errors.
    * One way to fix them is to uninstall it and install a specific version that is known to work
           ```bash
           pip uninstall -y opencv-python-headless
           pip install opencv-python==4.8.0.74
           ```
    * Another (hacky) way is to comment one line of the opencv code to suppress the error. The error looks like this
`File "/usr/local/lib/python3.10/dist-packages/cv2/typing/__init__.py", line 169, in <module>
    LayerId = cv2.dnn.DictValue
AttributeError: module 'cv2.dnn' has no attribute 'DictValue'`
          It can be fixed by commenting line 169 with the editor you prefer. In vim it looks like `vi /usr/local/lib/python3.10/dist-packages/cv2/typing/__init__.py; :169; i; #; esc; :wq; ` (; = press enter on the keyboard).

So if you want it to work on GPU you need to compile it from source with FFMPEG (also FFMPEG needs to be compiled for GPU!). To make opencv work with ffmpeg it required a few steps:
    * download manually the package from [github](https://github.com/opencv/opencv/releases/tag/4.7.0)
    * follow installation steps [here](https://gist.github.com/yjxiong/d716c857258f0295b58d148fbf8c489d)
    * remember to remove the [CMakeCache.txt](https://stackoverflow.com/questions/45518317/in-source-builds-are-not-allowed-in-cmake)

Check with this command if opencv supports ffmpeg `print(cv2.getBuildInformation())`

*NOTE:* it could happen you have errors with FFMPEG that does not find the CUDA Video library. To solve this, plese download NVIDIA Video Codec [here](https://developer.nvidia.com/nvidia-video-codec-sdk/download) it will ask you to login probably. While running the docker copy `libnvcuvid.so` under CUDA (e.g. in my case `/usr/local/cuda/lib64/.`).

## Getting Started

Start learning about video from the [00 Video Data Handling](00%20Video%20Data%20Handling.ipynb) notebook.
Then move on to the video libraries tests in [01 Video Libraries Tests](01%20Video%20Libraries%20Tests.ipynb)

To visualize and run the notebooks one option is to run the container and then run jupyter from there (tested and maintained). But feel free to use the way you prefer.

```bash
git clone https://github.com/NVIDIA-AI-Technology-Center/accelerated-video-for-ai-playbook.git

docker run --gpus all -it --rm -v /path/to/github/project/:/workspace --shm-size=8g -p 8888:8888 --network host --ulimit memlock=-1 --ulimit stack=67108864 nvcr.io/nvidia/pytorch:25.03-py3

cd accelerated-video-for-ai-playbook

jupyter notebook
```

## Disclaimer

The codebase might be subject to further changes as new versions of used libraries become available or new functionalities requested. The video libraries tests are a best effort and not to be considered a professional benchmark.
