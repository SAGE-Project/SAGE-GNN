# GNN-Cloud-Deployment

## Table of Contents

- [Introduction](#introduction)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [License](#license)


## Introduction

With the rising importance of Cloud deployment, organizations face the intricate task of optimally deploying component-based applications on diverse Virtual Machine (VM) offerings. While robust solutions like Kubernetes and AWS Elastic Beanstalk exist, they don't target this specific challenge efficiently.

This project introduces a unique approach combining Graph Neural Networks (GNNs) and the SMT solver Z3. Leveraging GNNs' capability to interpret graph-structured data, we model past deployments as graphs, enabling the prediction of optimal VM assignments based on historical data.

By using these GNN-based predictions as soft constraints in Z3, we enhance search efficiency, making the deployment process both more efficient and cost-effective.


## Features

1. **Dataset Generation:** 
   - Generate a dataset to train the GNN model for the application deployment.
   - For a detailed look into the data generation process: 🔗 [src/generate_dataset.py](./src/generate_dataset.py)

2. **GNN Model Implementation:**
   - Construct and train the GNN model able to predict component-to-VM assignments and VM Offer types.
   - Save trained model for future use.
   - Explore the implementation: 🔗 [src/gnn.py](./src/trainRGCN.py)
   - 🔗 Saved Model: [Models/GNNs/SecureWebContainer/model_RGCN_50_samples_100_epochs.pth](./Models/GNNs/SecureWebContainer/model_RGCN_50_samples_100_epochs.pth)

3. **Integration with SMT Solver Z3:**
   - Transform GNN predictions into soft constraints.
   - Guide the Z3 solver towards an optimal solution using these constraints.
   - See: 🔗 [src/Wrapper_GNN_Z3.py](./src/Wrapper_GNN_Z3.py)

## Installation

### 1. Clone the repository

```
git clone https://github.com/SAGE-Project/SAGE-GNN.git
cd SAGE-GNN
```

### 2. Setting Up a Conda Environment

- First, make sure you have [Anaconda](https://www.anaconda.com/products/distribution) or [Miniconda](https://docs.conda.io/en/latest/miniconda.html) installed on your system.

- **Create a new environment** (replace "myenv" with your desired environment name):

```bash
conda create --name myenv python=3.10
```

- **Activate the environment**:
```bash
conda activate myenv
```

### 3. Install Dependencies

This project relies on the following open-source libraries and tools:

- **Python (v3.10)**
- **Deep Graph Library (DGL v0.91)**
- **PyTorch (v1.13.0)**
- **CUDA (v11.6)**

Please ensure you have these dependencies installed and configured correctly before running the project.

## Usage

Using the already trained GNN models (from Models/GNNs/SecureWebContainer/), and the SecureWebContainer descriptions (from Models/json/) compare the results between:
   - Base, 
   - Base+FVPR, 
   - Base+GNN, 
   - Base+GNN+FVPR
   - See: 🔗 [src/comparison.py](./src/comparison.py)

## License

This project is licensed under the [BSD 3-Clause License](LICENSE).
