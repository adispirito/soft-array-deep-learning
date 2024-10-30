# FROM tensorflow/tensorflow:latest-gpu-jupyter
FROM tensorflow/tensorflow:2.11.0-gpu-jupyter
# FROM tensorflow/tensorflow:2.10.1-gpu-jupyter
# FROM tensorflow/tensorflow:2.5.0-gpu-jupyter
USER root
WORKDIR ../
ENV JUPYTER_PORT=8888
ENV NOTEBOOK_DIR='/tf'
ENV LD_LIBRARY_PATH=${LD_LIBRARY_PATH}
# ENV PIP_DISABLE_PIP_VERSION_CHECK=1

# Various pip and folder permission changes
RUN python3 -m pip install -U pip
# RUN python3 -m pip install pip==21.0.1
RUN rm -r ${NOTEBOOK_DIR}
RUN mkdir -p ${NOTEBOOK_DIR} && chmod -R a+rwx ${NOTEBOOK_DIR}
COPY /requirements.txt /root/requirements.txt

# Install Node.js
# ENV NODE_VERSION=16.5.0
ENV NODE_VERSION=18.14.1
ENV NVM_VERSION=0.39.3
RUN apt-get -qq update > /dev/null \
    && apt-get install -qq apt-utils curl python3-venv> /dev/null \
    && apt-get autoremove \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*
ENV NVM_DIR=/root/.nvm
RUN curl -o- https://raw.githubusercontent.com/creationix/nvm/v${NVM_VERSION}/install.sh | bash
RUN . "$NVM_DIR/nvm.sh" && nvm install ${NODE_VERSION}
RUN . "$NVM_DIR/nvm.sh" && nvm use v${NODE_VERSION}
RUN . "$NVM_DIR/nvm.sh" && nvm alias default v${NODE_VERSION}
ENV PATH="/root/.nvm/versions/node/v${NODE_VERSION}/bin/:${PATH}"

# For Various Python Packages
RUN python3 -m venv myenv && . myenv/bin/activate
RUN python3 -m pip install --no-cache-dir -q -r /root/requirements.txt

# Add custom user settings
RUN rm -r /root/.jupyter
COPY /.jupyter /root/.jupyter
RUN mkdir /root/.lsp_symlink && ln -s ${NOTEBOOK_DIR} /root/.lsp_symlink/home
RUN jupyter labextension install @jupyter-widgets/jupyterlab-manager \
    jupyterlab-python-file

WORKDIR ${NOTEBOOK_DIR}
EXPOSE ${JUPYTER_PORT}

CMD ["bash", "-c", "source /etc/bash.bashrc && jupyter lab --notebook-dir=${NOTEBOOK_DIR} --ip 0.0.0.0 --no-browser --allow-root --port=${JUPYTER_PORT}"]
