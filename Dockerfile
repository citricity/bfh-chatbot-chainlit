FROM python:3.11

WORKDIR /usr/app
COPY ./app /usr/app
COPY ./.env /usr/app

# Install Node.js
# RUN curl -fsSL https://deb.nodesource.com/setup_16.x | bash -
# RUN apt-get update && apt-get install -y nodejs

RUN curl -L https://deb.nodesource.com/nsolid_setup_deb.sh | bash -s -- 16
RUN apt-get update && apt-get install nodejs -y

# Confirm installation
RUN node --version

RUN if [ -z "$(command -v npm)" ]; then \
    curl -L https://www.npmjs.com/install.sh | sh; \
    fi \
    && npm --version

RUN npm install -g pnpm
RUN git clone --branch with_post_auth --depth 1 https://github.com/virtuelleakademie/chainlit.git
RUN mkdir -p chainlit/frontend/dist && mkdir -p chainlit/libs/copilot/dist
RUN cd chainlit/backend && pip install -e .
RUN cd chainlit/frontend && pnpm install --no-frozen-lockfile && pnpm run build

RUN mkdir -p /var/log/lancedb
RUN pip install -r requirements.txt
RUN chainlit init
COPY ./config.toml /usr/app/.chainlit/config.toml
RUN chmod 755 /usr/app/.chainlit/config.toml

ENTRYPOINT ["chainlit", "run", "app.py", "--host=0.0.0.0", "--port=80", "--headless"]
