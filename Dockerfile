# # ----- Build Stage -----
# FROM python:3.13 AS BUILD

# RUN apt-get update && \
#     apt-get install -y --no-install-recommends gcc python3-dev && \
#     rm -rf /var/lib/apt/lists/*

# COPY requirements.txt .
# RUN mkdir -p /install && \
#     pip install --target=/install -r requirements.txt

# # ----- Runtime Stage -----
# FROM python:3.13-slim AS FINAL

# # Copy installed packages
# COPY --from=BUILD /install /install
# ENV PYTHONPATH=/install

# COPY . .
# WORKDIR /app
# # CMD ["python", "your_script.py"]

FROM python:3.13-slim

# Copy the pre-built virtual environment
COPY ./venv /opt/venv

# Ensure the container uses the virtual environment
ENV PATH="/opt/venv/bin:$PATH"

# Copy your application code
COPY . /app
WORKDIR /app
