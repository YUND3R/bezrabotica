FROM python:3.9-slim
WORKDIR /app
COPY . .
RUN pip install --no-cache-dir -r requirements.txt
# Будет запускатся приложение градио в локальном формате
CMD ["python", "app.py"]
