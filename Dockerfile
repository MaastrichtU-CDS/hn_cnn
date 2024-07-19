# Original image used on the openshift platform:
# VS code: ghcr.io/maastrichtu-ids/code-server@sha256:56a27bb3171050a490a01e56b735c05fbfd96c3b6d0194e3837e0a3f99e99534
# Jupyter Notebook: ghcr.io/maastrichtu-ids/jupyterlab@sha256:08caa3920ba16eee04f2acf16b2f066f7a3e336cdc8f9cee3901a9e4ed256612
FROM python:3.8.8

# Install the dependencies
COPY ./requirements.txt /requirements.txt
RUN pip install -r requirements.txt
# Install the package
COPY ./hn_cnn/ /app/hn_cnn
COPY ./setup.py ./README.md /app/
RUN pip install --no-cache-dir /app
