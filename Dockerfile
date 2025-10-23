FROM harbor.irtsysx.fr/docker-proxy-cache/python:3.12

LABEL project="DataIA - tadkit"
LABEL version="1.0"
LABEL maintainer="Martin Royer"
LABEL description="dockerfile for streamlit"

WORKDIR /app

# Install pipx and uv
RUN python3 -m ensurepip && \
    pip install --break-system-packages --no-cache-dir pipx && \
    pipx install uv


# Add uv to PATH
ENV PATH="/root/.local/bin:$PATH"

RUN uv venv /venv

ENV VIRTUAL_ENV=/venv

ENV PATH="$VIRTUAL_ENV/bin:$PATH"

COPY . .
RUN uv pip install -r requirements.txt .
# RUN uv pip install --system -r requirements.txt .
RUN uv pip install git+https://github.com/IRT-SystemX/tdaad@opt

ENV PYTHONPATH=/app

EXPOSE 8501

RUN echo "alias ll='ls -lah --color=auto'" >> ~/.bashrc
CMD ["streamlit", "run", "visualiser/visualiser.py"]

