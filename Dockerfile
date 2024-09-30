FROM python:3.12

RUN pip install pandas torch pillow
RUN pip install tqdm torchvision