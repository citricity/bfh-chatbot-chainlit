FROM python:3.11

WORKDIR /usr/app
COPY ./app.py /usr/app
COPY ./.env /usr/app
COPY ./requirements.txt /usr/app
COPY ./rs256.rsa.pub /usr/app
COPY ./chainlit.md /usr/app

RUN pip install -r requirements.txt
RUN chainlit init
RUN sed -i "s/multi_modal = .*/multi_modal = false/" /usr/app/.chainlit/config.toml

ENTRYPOINT ["chainlit", "run", "app.py", "--host=0.0.0.0", "--port=80", "--headless"]
