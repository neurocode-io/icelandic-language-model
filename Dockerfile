FROM pytorch/pytorch:1.6.0-cuda10.1-cudnn7-runtime
WORKDIR /code

COPY requirements.txt .

RUN pip install --user -r requirements.txt

COPY src/ src/

# update PATH environment variable
ENV PATH=/root/.local:$PATH

CMD [ "python", "/code/01_fetch_data.py" ]

