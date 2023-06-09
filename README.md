# GlueFL Artifact

This repository contains the artifact for **GlueFL: Reconciling Client Sampling and Model Masking for Bandwidth Efficient Federated Learning** accepted at the *Sixth Conference on Machine Learning and Systems* (**MLSys 2023**). 

GlueFL is built as a component on top of the FedScale platform. The main GlueFL logic is located in the `./examples/gluefl` directory which contains both sticky sampling and mask shifting. GlueFL also adds several minor modifications to the `./fedscale` directory.

## Getting Started and Running Experiments
To run experiments using GlueFL, you should first set up FedScale following the standard [FedScale installation instructions](#quick-installation-linux).

After setting up FedScale, you can download the datasets used for experiments in our paper with the following commands. Note: you only need to download the datasets for the experiments that you want to run.

```shell
# To donwload the FEMNIST dataset (3400 clients, 640K samples, 327 MB)
fedscale download femnist 
# To donwload the Google Speech dataset (2618 clients, 105K samples, 2.3 GB)
fedscale download speech
# To donwload the Open Images dataset (13,771 clients, 1.3M samples, 66 GB)
fedscale download open_images
```

Since some of the datasets are quite large, you may want to download the dataset at another location. To do this, simply copy `./benchmark/dataset/download.sh` to the desired location and run the download command from there. If you do this, please take note of where you dowloaded the dataset. You will need to update your [configurations](#configurations) with that new location.

You should then be able to run experiments by supplying configurationf files. An example for running the GlueFL on the FEMNIST dataset with the ShuffleNet model is shown below.

```shell
fedscale driver start ./benchmark/configs/baseline/gluefl_femnist_shf.yml
```

## Configurations

In GlueFL, every experiment has their own configuration YAML file containg all the settings related to that experiment. You can find all the experiment configurations used in our paper in the `./benchmark/configs` directory. There are four sub-directories corresponding to different sections of our paper.

- `./benchmark/configs/baseline` contains the experiment configurations for near-optimal settings of GlueFL, FedAvg, STC, and APF. The results from these experiment runs are used to fill Table 2
- `./benchmark/configs/sensitivity` contains the experiment configurations used for the sensitivity analysis by differing GlueFL's hyper-parameters on the FEMNIST and Google Speech data sets.
- `./benchmark/configs/ablation` contains the experiment configurations used for the ablation study by differing other settings (reweighting and error compensation) on the FEMNIST and Google Speech datasets.
- `./benchmark/configs/environment` contains the experiment configurations used for the network environment study by swapping different client device profiles. 

### Notes:
- **If you downloaded your dataset to a different location**, please update the `data_dir` and `data_map_file` settings in the configuration file.

- You may want to specify a different location for the `compensation_dir` which is used to store client-side error compensation data because they tend to get quite large (at least 40 GB). **Remember to periodically delete your compensation_dir after finishing experiments to release storage space!**

- You can run experiments with just CPUs by setting `use-cuda` to `False`

- Although you can use multiple GPUs on a single machine (e.g. `benchmark/configs/baseline/fedavg_image_shf.yml`), you should not try to use multiple machines to run experiments. We are working towards removing this limitation.

## Viewing Results

You can view the results for an experiment by using the following commands with the generated log file in the project root directory.

```shell
# To view the training loss
cat job_name_logging | grep 'Training loss'
# To view the top-1 and top-5 accuracy
cat job_name_logging | grep 'FL Testing'
# To view the current bandwidth usage and training time
cat job_name_logging | grep -A 9 'Wall clock:'
# To view the bandwidth usage and training time of a particular round (for example, 500)
cat job_name_logging | grep -A 9 'round: 500'
```

You can also find logs just for the aggregator and executor in the directory specified by the `log_path` setting.


<br/>

---

<br/>
<br/>


<p align="center">
<img src="./docs/imgs/FedScale-logo.png" width="300" height="55"/>
</p>

[![](https://img.shields.io/badge/FedScale-Homepage-orange)](https://fedscale.ai/)
[![](https://img.shields.io/badge/Benchmark-Submit%20Results-brightgreen)](https://fedscale.ai/docs/leader_overview)
[![](https://img.shields.io/badge/FedScale-Join%20Slack-blue)](https://join.slack.com/t/fedscale/shared_invite/zt-uzouv5wh-ON8ONCGIzwjXwMYDC2fiKw)

**FedScale is a scalable and extensible open-source federated learning (FL) engine and benchmark**. 

FedScale ([fedscale.ai](https://fedscale.ai/)) provides high-level APIs to implement FL algorithms, deploy and evaluate them at scale across diverse hardware and software backends. 
FedScale also includes the largest FL benchmark that contains FL tasks ranging from image classification and object detection to language modeling and speech recognition. 
Moreover, it provides datasets to faithfully emulate FL training environments where FL will realistically be deployed.


## Getting Started

### Quick Installation (Linux)

You can simply run `install.sh`.

```
source install.sh # Add `--cuda` if you want CUDA 
pip install -e .
```

Update `install.sh` if you prefer different versions of conda/CUDA.

### Installation from Source (Linux/MacOS)

If you have [Anaconda](https://www.anaconda.com/products/distribution#download-section) installed and cloned FedScale, here are the instructions.
```
cd FedScale

# Please replace ~/.bashrc with ~/.bash_profile for MacOS
FEDSCALE_HOME=$(pwd)
echo export FEDSCALE_HOME=$(pwd) >> ~/.bashrc 
echo alias fedscale=\'bash $FEDSCALE_HOME/fedscale.sh\' >> ~/.bashrc 
conda init bash
. ~/.bashrc

conda env create -f environment.yml
conda activate fedscale
pip install -e .
```

Finally, install NVIDIA [CUDA 10.2](https://developer.nvidia.com/cuda-downloads) or above if you want to use FedScale with GPU support.


### Tutorials

Now that you have FedScale installed, you can start exploring FedScale following one of these introductory tutorials.

1. [Explore FedScale datasets](./docs/Femnist_stats.md)
2. [Deploy your FL experiment](./docs/tutorial.md)
3. [Implement an FL algorithm](./examples/README.md)


## FedScale Datasets

***We are adding more datasets! Please contribute!***

FedScale consists of 20+ large-scale, heterogeneous FL datasets covering computer vision (CV), natural language processing (NLP), and miscellaneous tasks. 
Each one is associated with its training, validation, and testing datasets. 
We acknowledge the contributors of these raw datasets. Please go to the `./benchmark/dataset` directory and follow the dataset [README](./benchmark/dataset/README.md) for more details.

## FedScale Runtime
FedScale Runtime is an scalable and extensible deployment as well as evaluation platform to simplify and standardize FL experimental setup and model evaluation. 
It evolved from our prior system, [Oort](https://github.com/SymbioticLab/Oort), which has been shown to scale well and can emulate FL training of thousands of clients in each round.

Please go to `./fedscale/core` directory and follow the [README](./fedscale/core/README.md) to set up FL training scripts.


## Repo Structure

```
Repo Root
|---- fedscale          # FedScale source code
  |---- core            # Core of FedScale service
  |---- utils           # Auxiliaries (e.g, model zoo and FL optimizer)
  |---- deploy          # Deployment backends (e.g., mobile)
  |---- dataloaders     # Data loaders of benchmarking dataset

|---- benchmark         # FedScale datasets and configs
  |---- dataset         # Benchmarking datasets

|---- examples          # Examples of implementing new FL designs
|---- docs              # FedScale tutorials and APIs
```

## References
Please read and/or cite as appropriate to use FedScale code or data or learn more about FedScale.

```bibtex
@inproceedings{fedscale-icml22,
  title={{FedScale}: Benchmarking Model and System Performance of Federated Learning at Scale},
  author={Fan Lai and Yinwei Dai and Sanjay S. Singapuram and Jiachen Liu and Xiangfeng Zhu and Harsha V. Madhyastha and Mosharaf Chowdhury},
  booktitle={International Conference on Machine Learning (ICML)},
  year={2022}
}
```

and  

```bibtex
@inproceedings{oort-osdi21,
  title={Oort: Efficient Federated Learning via Guided Participant Selection},
  author={Fan Lai and Xiangfeng Zhu and Harsha V. Madhyastha and Mosharaf Chowdhury},
  booktitle={USENIX Symposium on Operating Systems Design and Implementation (OSDI)},
  year={2021}
}
```

## Contributions and Communication
Please submit [issues](https://github.com/SymbioticLab/FedScale/issues) or [pull requests](https://github.com/SymbioticLab/FedScale/pulls) as you find bugs or improve FedScale.

If you have any questions or comments, please join our [Slack](https://join.slack.com/t/fedscale/shared_invite/zt-uzouv5wh-ON8ONCGIzwjXwMYDC2fiKw) channel, or email us ([fedscale@googlegroups.com](mailto:fedscale@googlegroups.com)). 

