# Use an official Python base image and install NodeJS via apt
FROM python:3.10-slim AS base

# Avoid python buffering stdout/stderr
ENV PYTHONUNBUFFERED=1

# Create application directory
WORKDIR /app

# Copy Python requirements and install
COPY requirements.in ./


#RUN pip install --no-cache-dir  --upgrade pip
#RUN pip install   --upgrade pip
RUN pip install pip-tools
RUN pip-compile requirements.in --output-file=./requirements.txt

#RUN pip install --no-cache-dir -r requirements.txt
RUN pip install  -r requirements.txt

# Copy Python source code
COPY app ./app/

# Expose Streamlit default port
EXPOSE 8501
EXPOSE 8502

# Set PYTHONPATH so the app modules can be found
ENV PYTHONPATH=/app/app

# Run the Streamlit app
CMD ["streamlit", "run", "app/streamlit_app.py", "--server.port", "8501", "--server.address", "0.0.0.0"]
