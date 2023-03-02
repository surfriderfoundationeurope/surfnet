FROM cnstark/pytorch:1.13.1-py3.9.12-ubuntu20.04

ARG DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y wget gcc git ffmpeg libsm6 libxext6

ADD requirements.txt .

RUN pip install -r requirements.txt && \
  rm requirements.txt

ADD data/tracking_parameters/ /serve/data/tracking_parameters

RUN mkdir -p /serve/models/

RUN wget https://github.com/surfriderfoundationeurope/surfnet/releases/download/v01.2023/yolo_latest.pt && mv yolo_latest.pt /serve/models/yolov5.pt

ADD scripts/serving_prod.sh /scripts/serving_prod.sh

ENTRYPOINT /scripts/serving_prod.sh