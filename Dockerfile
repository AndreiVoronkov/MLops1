FROM python:3.12.4
# RUN pip install fastapi uvicorn numpy
# RUN pip install joblib 
# RUN pip install catboost 
# RUN pip install scikit-learn
# RUN pip install clearml
WORKDIR /app
# COPY . /app/
# COPY ./* /

# COPY pyproject.toml ./
COPY pyproject.toml ./

RUN python -m pip install --no-cache-dir poetry==1.6.1 
    # && poetry config virtualenvs.create false \
    # && poetry install --without dev --no-interaction --no-ansi \
    # && rm -rf $(poetry config cache-dir)/{cache,artifacts}

COPY . /app/
# WORKDIR /app
    # && poetry config virtualenvs.create false \
    # && poetry install --without dev --no-interaction --no-ansi \
    # && rm -rf $(poetry config cache-dir)/{cache,artifacts}

# RUN -p 8000:8000 \
#     -e "MINIO_ROOT_USER=myaccesskey" \
#     -e "MINIO_ROOT_PASSWORD=mysecretkey" \
#     -v /path/to/data:/data \
#     -d minio/minio server /data

# RUN clearml-init --file clearml.conf
# ENV CLEARML_CONFIG_FILE clearml.conf

CMD uvicorn api:app --reload --host 0.0.0.0
# CMD ["poetry", "run"]
# docker build . -t my-service -- развернуть сервер
# docker run -p 8000:8000 my-service -- для запуска докера
# http://127.0.0.1:8000/docs -- сайт на который нужно зайти
# http://127.0.0.1:9000/docs -- сайт для minio s3


# ----------------------------------
# Установить pyproject.toml
# Поэтапно прописываем в командной строке
#  1. pip install poetry
#  2. poetry init 
#  3. poetry lock
#  4. poetry install


# Послдовательность действий для особо умных (то есть для меня):
# 1. Начинаем с pip install poetry -> poetry init (вообще лучше прописать вручную)
# 2. Благодаря строке poetry lock появляется папка poetry.lock
# 3. poetry install накатываем зависимости
# 4. затем запускаем docker build . -t my-service
# 5. Потом docker run -p 8000:8000 my-service

