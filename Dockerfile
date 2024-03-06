FROM python:3.11

WORKDIR /usr/app
COPY ./app /usr/app
COPY ./.env /usr/app

RUN mkdir -p /var/log/lancedb
RUN pip install -r requirements.txt
RUN chainlit init
COPY ./config.toml /usr/app/.chainlit/config.toml
RUN chmod 755 /usr/app/.chainlit/config.toml

ENTRYPOINT ["chainlit", "run", "app.py", "--host=0.0.0.0", "--port=80", "--headless"]
