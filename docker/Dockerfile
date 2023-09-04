FROM pytorch/pytorch:1.12.1-cuda11.3-cudnn8-runtime
RUN apt update
RUN apt-get update
RUN apt install -y openslide-tools
RUN apt-get install -y build-essential libgl1-mesa-dev
RUN python -m pip install scikit-learn pandas matplotlib openslide-python scikit-image timm opencv-python albumentations jupyter seaborn