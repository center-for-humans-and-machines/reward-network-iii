# Reward Networks

This repository contains the code for the project "Experimental Evidence for the Propagation and Preservation of Machine Discoveries in Human Populations" 

This repository allows to:
- Train a machine player to solve reward networks
- Run an online experiment where human participants solve reward networks task
- Analyze the data from the experiment
- Visualize the results

This repository contains both the code and the experimental data.

## Overview of Resources

### Data

The data is stored in the `data` directory. The data is structured as follows:
- `data/networks_solutions_models` contains the networks, trained neural network models, and solutions for the networks (both, from the neural networks and three prototypical heuristic strategies).
- `data/exp_raw` contains the raw data from the experiment as downloaded from the online experiment.
- `data/exp_processed` contains the processed data from the experiment, including the alignment between human and machine actions and written strategies.
- `data/exp_strategies_coded` contains the manually coded written strategies.
- `data/abm` contains the data from the agent-based model (after running the corresponding notebook).

### Algorithm

The algorithm is implemented in the [algorithm](algorithm) directory. The algorithm trains a neural policy to solve reward networks tasks.

### Online Experiment

The online experiment is hosted on the [backend](backend) and [frontend](frontend) services. The frontend is a React application that allows participants to solve reward networks tasks. The backend is a Flask application that serves the frontend and stores the data from the experiment.

### Visualizations

The visualizations are stored in the `analysis/plots` directory. The corresponding notebooks are stored in the `analysis` directory.

### Statistical Analysis

The statistical analysis is stored in the `statistics` directory. See also the respective README file therein.

## Installation

### System Requirements

This software has been tested with `python3.10`. Required packages can be installed using setup.py (see below). Tested package versions are defined in the respective requirements files within the folder `setup/requirements`. Installation typically requires not more than a few minutes.

### Docker

Network generation, training of the algorithm, and running the backend can be done using Docker. To build the Docker image, run the following command:

```bash
docker compose build all
```

### Local Environment

For running the descriptive analysis of the results, the following setup is required:

```bash
python3.10 -m venv .venv
. .venv/bin/activate
pip install --upgrade pip
pip install wheel
pip install -e ".[viz]"
```

Or us the following command to install all dependencies:

```bash
pip install -e ".[viz,dev,backend,train]"
```

## Demo and Usage

### Training of the Machine Player

We provide example commands for running the training. Depending on the hardware, this process can take several hours to complete.
See the respective [algorithm README](algorithm/README.md) for more details.

### Run the experiment

Start the frontend and backend services using the following command:

```bash
docker compose up frontend backend
```

A static verion of the experiment is avaible here: https://center-for-humans-and-machines.github.io/reward-network-iii


### Visualizations of Experimental Data, and Algorithmic Learning Curve

Visualisations are produced in Jupyther Notebooks. These typically run within minutes. 
See the respective [analysis README](analysis/README.md) for more details.

### Agent-Based Model

The agent based model is defined in a Jupyther Notebook. These typically run within minutes.
See the respective [abm README](abm/README.md) for more details.

### Statistics

We use an R script to perform the statistical analyses. Running the script can take more than five hours due to the bootstrapping procedure. To reduce runtime, one might decrease the number of bootstrap samples or deactivate the boostrapping all together.
See the respective [statistics README](statistics/README.md) for more details.