FROM pytorch/pytorch:latest

RUN apt-get update && apt-get install -y wget gcc git ffmpeg libsm6 libxext6

ADD requirements.txt .

RUN pip install -r requirements.txt && \
  rm requirements.txt

ADD models /models

RUN chmod +x  /models/download_pretrained_base.sh


ADD data/tracking_parameters/ /serve/data/tracking_parameters

RUN cp -r /models /serve/models

RUN /models/download_pretrained_base.sh \
&& mv mobilenet_v3_pretrained.pth /serve/models/



ADD scripts/serving_prod.sh /scripts/serving_prod.sh

ENTRYPOINT /scripts/serving_prod.sh








