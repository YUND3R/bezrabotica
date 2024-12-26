# Пояснения

## Сборка Docker-образа:

```
docker build -t yund3r/bezrabotica:latest .
```

## Запуск Docker-контейнера:

```
docker run -it --rm -p 5000:5000 yund3r/bezrabotica:latest
```

--- 

## Загрузка образа на Docker Hub:

После регистрации выполни следующие команды для входа в Docker Hub и загрузки образа:

```
docker login
docker push yund3r/bezrabotica:latest
```
