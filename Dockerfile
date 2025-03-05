FROM  xychelsea/anaconda3:latest-gpu-jupyter
WORKDIR $HOME/molclr
COPY . .
USER root
RUN chown -R ${ANACONDA_USER}:${ANACONDA_GID} ${HOME}/molclr
RUN chmod -R 755 .
USER anaconda
RUN conda update -n base conda
RUN pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu124
RUN pip install torch_geometric PyYAML scikit-learn
RUN pip install torch-scatter torch-sparse -f https://data.pyg.org/whl/torch-2.5.1+$cu124.html
RUN conda install -c conda-forge rdkit=2022.3.3
RUN conda install -c conda-forge tensorboard
USER anaconda
