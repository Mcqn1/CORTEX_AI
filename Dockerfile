
FROM python:3.9


WORKDIR /app


COPY . /app

RUN pip install --no-cache-dir -r requirements.txt


RUN pip install dvc[gdrive]


EXPOSE 5000

CMD ["python", "EEG_TRAIN_MODEL.py"]
