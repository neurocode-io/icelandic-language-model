# first stage
FROM python:3.7 AS builder
COPY requirements.txt .

# install dependencies to the local user directory (eg. /root/.local)
RUN pip install --user -r requirements.txt

# second unnamed stage
FROM pytorch/pytorch:1.6.0-cuda10.1-cudnn7-runtime
WORKDIR /code

# copy only the dependencies installation from the 1st stage image
COPY --from=builder /root/.local/bin /root/.local
COPY . .

# update PATH environment variable
ENV PATH=/root/.local:$PATH

CMD [ "python", "./language_model/train_tokenizer.py" ]

