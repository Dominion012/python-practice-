FROM python:3.11-slim

WORKDIR /app

COPY requirements-render.txt .
RUN pip install -r requirements-render.txt

COPY day69.py .

EXPOSE 8027

CMD ["uvicorn", "day69:app", "--host", "0.0.0.0", "--port", "8027"]
