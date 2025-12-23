FROM python:3.10-slim

WORKDIR /app

COPY model/model.pkl /app/model.pkl
COPY requirements.txt .

RUN pip install -r requirements.txt

CMD ["python"]
