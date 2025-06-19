# Auto ru prediction cars price

This repository contains a code for final project for [nlp course](https://ods.ai/tracks/nlp-course-spring-2025) conducted by Valentin Malykh (MTS AI)


The main goal of the project is to make a system for predicting the price of a car based on the following data (year of manufacture, model, description from the seller, characteristics, etc.). Full dataset description can be found 
in report folder. 

## How launch code for this project

First step is a clone project repository

```bash
git clone https://github.com/infamax/hse_nlp_project.git
```

Second step is launch docker container for this project. Docker container contains all needed dependencies for this project. To launch docker container for this project. Assume you have gpu and install [nvidia-container-toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html). Run the following commands

```bash
make build_image # Create docker image for this project
make run_container # Run docker container
make attach_container # Attach a running container for this project
```

Third step is collect dataset from [auto.ru](https://auto.ru/) site. Run the following commands

```bash
python3 scripts/parser.py --destination <path_to_dataset_folder>
python3 scripts/split_dataset.py --dataset-folder <path_to_dataset_folder>
```

<path_to_dataset_folder> - Any folder you want to collect dataset.



