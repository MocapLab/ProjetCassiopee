# 📝 Notes

## 🛠️ Set-up
- Guide: https://vulkan.telecom-sudparis.eu/help/
- Launch Python virtual environment wizard:
```bash
/space/tools/scripts/setupFramework.sh
```
- Select:
  - Virtualenv
  - self_supervised_learning
  - [keep default]
  - Basic
  - Tested
  - Python 3.8.15
  - GPU
  - CUDA 11.5
- Activate Python virtual environment:
```bash
cd /home/self_supervised_tmp/self_supervised_learning
source start
```
- Set-up project:
```bash
# Set-up Git project
mkdir dev
cd dev
sudo apt install -y git
git config --local user.name "FABallemand"
git config --local user.email "allemand.fabien@orange.fr"
git clone https://github.com/FABallemand/ProjetCassiopee
cd ProjetCassiopee

# Download utility
sudo apt install htop

# Download data
pip3 install gdown
sudo apt-get install -y p7zip-full p7zip-rar
nohup ./data/RGB-D_Object/download.sh &
nohup ./data/mocaplab/download.sh &

# Set-up Jupyter
# https://janakiev.com/blog/jupyter-virtual-envs/#add-virtual-environment-to-jupyter-notebook
# Deactivate virtual environment?
pip3 install --user ipykernel
python3 -m ipykernel install --user --name=self_supervised_learning
jupyter kernelspec list

# Download requirements
pip3 install -r requirement.txt
pip3 install torch==1.12.1+cu113 torchvision==0.13.1+cu113 torchaudio==0.12.1 --extra-index-url https://download.pytorch.org/whl/cu113
```
- https://pytorch.org/blog/deprecation-cuda-python-support/
- https://pytorch.org/get-started/previous-versions/

## 💾 Useful Commands

### Data
```bash
tar -xvf file.tar
tar -xzvf file.tar.gz

7za x file.zip -ooutput/path

# Download Google Drive folder/file
gdown --folder link/to/folder -O /path/to/dir

# Copy folder/file from local to server
scp -r /path/to/local/dir user@remotehost:/path/to/remote/dir
```

### Python
```bash
# Run program with nohup (see also tmux)
nohup python3 -u main.py &
```

### Debugging
```bash
# Kill nohup process
# https://stackoverflow.com/questions/17385794/how-to-get-the-process-id-to-kill-a-nohup-process

# Run program with CUDA_LAUNCH_BLOCKING=1
CUDA_LAUNCH_BLOCKING=1 nohup python3 -u main.py &

# Reboot server
sudo reboot now

# Clear RAM cache
sudo sync; echo 1 > /proc/sys/vm/drop_caches

# Kill all process using pattern
pkill -f <pattern>

# "Restart" GPU on Ubuntu
# https://discuss.pytorch.org/t/cuda-fails-to-reinitialize-after-system-suspend/158108/3
sudo rmmod nvidia_uvm
sudo modprobe nvidia_uvm

# Debugging tool
dmesg | grep GPU

# Read end of file
# https://kb.iu.edu/d/acrj
tail [ +-[number][lbcr] ] [file]
```

## 🗂️ Dataset
- [RGBD-SOD Dataset](https://www.kaggle.com/datasets/thinhhuynh3108/rgbdsod-set1)
- [How2Sign (A Large-scale Multimodal Dataset for Continuous American Sign Language)](https://paperswithcode.com/dataset/how2sign)
- [HIC (Hands in Action)](https://paperswithcode.com/dataset/hic)
- [RGB-D Object Dataset](https://rgbd-dataset.cs.washington.edu/dataset.html)
- [List of RGBD datasets](http://www.michaelfirman.co.uk/RGBDdatasets/)
- [Mocaplab Google Drive](https://drive.google.com/drive/folders/1xos4pybtOfltFU0_YPhWLEyWKH5h6SoO)
- [Motion-X: A Large-scale 3D Expressive Whole-body Human Motion Dataset](https://motion-x-dataset.github.io/)

## 🖥️ Code
- [Tmux Tutorial Video](https://www.youtube.com/watch?v=Yl7NFenTgIo&ab_channel=HackerSploit)
- [Tmux Cheat Sheet](https://tmuxcheatsheet.com/)
- [PyTorch v1.12 Documentation](https://pytorch.org/docs/1.12/)
- [PyTorch Tutorials](https://github.com/yunjey/pytorch-tutorial/tree/master)
- [SimCLR Article](https://arxiv.org/pdf/2002.05709.pdf)
- [SimCLR in PyTorch](https://medium.com/the-owl/simclr-in-pytorch-5f290cb11dd7)
- [SimCLR Tutorial](https://deeplearning.neuromatch.io/tutorials/W3D3_UnsupervisedAndSelfSupervisedLearning/student/W3D3_Tutorial1.html)
- [SimCLR Tutorial Videos](https://www.youtube.com/playlist?list=PLkBQOLLbi18NYb71nfD5gwwnZY4DPMCXu)
- [Demystify RAM Usage in Multi-Process Data Loaders](https://ppwwyyxx.com/blog/2022/Demystify-RAM-Usage-in-Multiprocess-DataLoader/)

## 🧠 AI
- [MMAction](https://github.com/open-mmlab/mmaction2)
- [MMAction graph-based action recognition](https://github.com/open-mmlab/mmaction2/blob/main/configs/skeleton/2s-agcn/README.md)
- [Two-Stream Adaptive Graph Convolutional Networks for Skeleton-Based Action Recognition](https://openaccess.thecvf.com/content_CVPR_2019/papers/Shi_Two-Stream_Adaptive_Graph_Convolutional_Networks_for_Skeleton-Based_Action_Recognition_CVPR_2019_paper.pdf)

## 📋 ToDo
- [] ...
