FROM ubuntu:focal
ARG DEBIAN_FRONTEND=noninteractive
#RUN export DEBIAN_FRONTEND=noninteractive



RUN apt-get update


RUN apt-get -yq install python3.9
RUN apt-get -yq install python3-pip
RUN apt-get -yq install python3-opencv
RUN apt-get -yq install libpng-dev
RUN apt-get -yq install libjpeg-dev
RUN apt-get -yq install libopenexr-dev
RUN apt-get -yq install libtiff-dev
RUN apt-get -yq install libwebp-dev
RUN apt-get -yq install tesseract-ocr-hin
RUN apt-get -yq install tesseract-ocr-tam
RUN apt-get -yq install tesseract-ocr-kan
RUN apt-get -yq install tesseract-ocr-tel
RUN apt-get -yq install nginx
RUN apt-get -yq install vim
RUN apt-get -yq install curl


#COPY requirements.txt .
#COPY templates/ templates/
#COPY serve.sh serve
#COPY train train
#COPY wsgi.py .

#COPY hin.traineddata . 

RUN ls -altrh

COPY ocr_1.0 /opt/program

WORKDIR /opt/program


RUN pip3 install -r requirements.txt

#COPY nginx.conf .
ENV PYTHONUNBUFFERED=TRUE
ENV PYTHONDONTWRITEBYTECODE=TRUE
ENV PATH="/opt/program:${PATH}"

RUN echo $PATH



RUN chmod +x serve
#RUN chmod +x train




#CMD python3 -m flask run --host=0.0.0.0 --port=80

