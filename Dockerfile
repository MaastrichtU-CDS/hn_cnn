FROM python:3.8.8

# Install the dependencies
COPY ./requirements.txt /requirements.txt
RUN pip install -r requirements.txt
# Install the package
COPY ./hn_cnn/ /app/hn_cnn
COPY ./setup.py ./README.md /app/
RUN pip install --no-cache-dir /app
