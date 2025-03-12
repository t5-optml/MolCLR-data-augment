# MolCLR Data Augmentation

## Getting Started

### Requirements

- Docker

### Getting started

Cloning the repository and the submodule

```bash
git clone --recursive https://github.com/t5-optml/MolCLR-data-augment.git
```

### Dataset

The submodule of [Myopic MCES Data](https://github.com/boecker-lab/myopic-mces-data) has been included as the dataset for us to use for pre-training.

The original dataset can be found [here](https://drive.google.com/file/d/1aDtN6Qqddwwn2x612kWz9g0xQcuAtzDE/view?usp=sharing). Simply download it and extract the zip file under `./data` folder. The data for pre-training can be found in `pubchem-10m-clean.txt`. All the databases for fine-tuning are saved in the folder under the benchmark name. You can also find the benchmarks from [MoleculeNet](https://moleculenet.org/).

You can also run the `sample.py` with your custom setting to sample the `pubchem-10m-clean.txt` dataset (thanks to Viet). However, we will use `myopic-mces-data` for our baseline for now.

### Installation

Run the following command to build the Docker image (might take a while, grab your snack and drink and watch an anime episode)

```bash
docker build -t t5-optml/molclr-data-augment .
```

Alternatively, if the dataset for the downstream task uses an older version of the `rdkit` library, run the following command to build the Docker image with an older version of `rdkit`

```bash
docker build -t t5-optml/molclr-data-augment -f Dockerfile.fix .
```

### Running

After that run the following command to run the imagee into a container. This will spin up a Jupyter Lab for you at port 8888 for you to access with your browser

```bash
docker run --gpus all --rm -it -d -p 8888:8888 t5-optml/molclr-data-augment
```

Copy the container ID that the above command return. If for some reason you lost that command, simply use Docker Desktop or run the following command

```bash
docker ps
```

And look for the container ID that host the image `t5-optml/molclr-data-augment`.

Now run the following command to receive your Jupyter Lab URL, where `{container-id}` is replaced with your container id.

```bash
docker logs {container-id}
```

Look for the section that said the following:

```bash
    To access the server, open this file in a browser:
        file:///home/anaconda/.local/share/jupyter/runtime/jpserver-7-open.html
    Or copy and paste one of these URLs:
        http://0e6e168f2d73:8888/lab?token=a762ae6c2334208ce7a3f7fb81951a628382ea9a8681b20d
        http://127.0.0.1:8888/lab?token=a762ae6c2334208ce7a3f7fb81951a628382ea9a8681b20d
```

Use the link with `http://127.0.0.1:8888` to access the Jupyter Lab from your browser.

In your Jupyter lab, create a new terminal window and run the following command to start training (thanks Viet for the conversion). This will use the `config.yaml` file as the base configuration for trainning.

```bash
python molclr_torch_amp.py
```

After you have pretrained your model, modify your `config_finetune.yaml` to take in the model from `ckpt` folder and take the task you want to finetune. Then run the following command

```bash
python finetune.py
```

An example of configuration is in the file `config.example.yaml`.

## Acknowledgement

- MolCLR: [https://github.com/yuyangw/MolCLR/](https://github.com/yuyangw/MolCLR/)
- Structure visualization via UMAP embedding of MCES distances [https://github.com/boecker-lab/myopic-mces-data](https://github.com/boecker-lab/myopic-mces-data)
