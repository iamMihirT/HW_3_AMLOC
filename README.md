# HW_3_AMLOC

This repository contains the code and configuration files for COMS 6998 HW3: Deployment of a Deep Learning Training and Inference Service on Google Kubernetes Engine (GKE).

## Overview

In this project, I developed container and Kubernetes artifacts to create a scalable, maintainable service for training a stock prediction model and serving inference requests. The project uses:
- **PyTorch** for model training.
- **Flask** for the inference API.
- A **Kubernetes Job** to execute the training process.
- A **Kubernetes Deployment** and **LoadBalancer Service** to manage and expose the inference application.
- A **PersistentVolumeClaim (PVC)** to share the trained model files between the training job and the inference service.

All source code, Dockerfiles, and Kubernetes manifest files are available in this repository.


