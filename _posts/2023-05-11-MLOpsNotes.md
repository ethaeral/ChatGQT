---
title: Machine Learning Ops Notes
date: 2023-05-11 00:00:00 -0500
categories: [Core]
tags: [forme, interview prep, notes]
---

# Principles of MLOps
## Versioning 
### Data and Model Versioning
Replicating same code so multiple people can work on the same project and go back into older versions of model is crucial for teams to work on and evaluate experiments.
### Reproducibility
It's important to make sure the key factors of a model are documented and reusable
- Assumptions
    - Decisions and assumptions must be explicit
- Randomness
    - Considering that some machine learning experiments contain pseudo-randomness, this randomness needs to be reproduced, like using seeds
- Data
- Settings
- Implementations
- Environment - run time configurations
### Popular Versioning Tool
| Tool | License | Developer | Observations |
|---|---|---|---|
| IBM Watson ML | Proprietary | IBM | Focused on model versioning |
| DVC | Open-source | Iterative | Popular lightweight open-source tool focused on data, model and pipeline versioning. Can be easily integrated with CML. |
| Pachyderm | Open-source | Pachyderm | Data platform built on Docker and Kubernetes. Focused on Version Control and automating workloads with parallelization. |
| MLflow | Open-source | Databricks | Popular tool for many parts of the Machine Learning lifecycle, including versioning of processes and experiments. Can be easily integrated with other tools |
| Git LFS (Large File System) | Open-source | Atlassian, GitHub, and others | Git extension that permits large files on Git repositories. Can be used to share large data files and models, using Git versioning. |

## Automation 
Automation will result in faster deployments, more reliable process, easier problem discovering.

Task that can be automated
| ML Stages | Tasks |
|---|---|
| Data Engineering | Data acquiring, validation, and processing |
| Model Development | Model training, evaluation, and testing |
| Continuous Integration | Build and testing |
| Continuous Delivery | Deployment new implementation of a model as a service |
| Monitoring | Setting alerts based on pre-defined metrics |

### Levels of Automation
- Manual Process: Full Experimentation pipeline executed manually using Rapid Application Development RAD tools, like Jupyter Notebooks. Deployments are also executed manually.
- Machine Learning Automation: Automation of the experiment pipeline which includes data and model validation
- CI/CD Pipelines: Automatically build, test, and deploy of ML models and ML training pipeline components, providing a fast and reliable deployment

### Common Tools for Machine
| Tools | License | Developer | Observations |
|---|---|---|---|
| DVC | Open-source | Iterative | DVC can be used to make Data Pipelines, which can be automated and reproduced. Very useful if already using DVC for Data and Model versioning. Easily configured and run. Language and framework agnostic. |
| Tensorflow Extended (TFX) | Open-source | Tensorflow | Used for production Machine Learning pipelines. Heavily integrated with Google and GCP. Only works with Tensorflow. |
| Kubeflow | Open-source | Google, Kubeflow | Kubeflow can build automated pipelines and experiments. Intended to build a complete end-to-end solution for Machine learning, being able to also serve and monitor models. Uses Kubernetes and is based on Tensorflow Extended. Works with Tensoorflow and Pytorch. |
| Kubeflow | Open-source | MLflow Project | Open-source platform for the machine learning lifecycle. Can be used with Python, Conda and Docker. Large community. |

## Feature Storing
The goal of Feature Store is to process data from various data sources at the same time and turn it into features, which will be consumed by the model training pipeline and the model serving.

### Significance
Useful for production to make reliable ways to manage features for research and training using offline stores.
#### Key Feature
- Enables features to be shared by multiple teams of Data Scientists working at the same time
- Creates a reliable automated preprocess pipeline of large quantities of data
- Can use and combine different data sources, such as data lakes, data warehouses and streaming of new data, all at once
- Provides relevant and online features to a model in production
- Can use a time windows system for Data Scientists to gather features from any point in time
- Highly customizable for different model needs of consumption, such as a batch or real-time predictions
### Offline Store vs Online Store
Main type of data 
- Batch Data: Usually coming from Data Lakes or Warehouses, not necessarily updated in real-time
- Real-time Data: Usually coming from Streaming and Log events

These types of data are combined inside and form two types of stores:
- Offline Stores
    - Composed of preprocessed features of Batch Dat
        - Used for building historical sources of features
        - Stored in data warehouses like IBM cloud object, APache Hive, S3
        - Databases PostgreSQL, Cassandra, MySQL, HDFS
- Online Stores
    - Composed of data from Offline Store Combined with real time preprocessed features from streaming data sources
        - Objective of being the most up to date collection of organized features
        - Stored in databases, MySQL, Cassandra, Redis

![Alt text](image-1.png)

### Popular Tools
| Tools | License | Developer | Observations |
|---|---|---|---|
| Michelangelo | Proprietary | Uber | Uber's platform for Machine Learning, focused on sharing feature pipelines with various teams. (Not open for public usage) |
| Zipline | Proprietary | AirBnB | Airbnb’s Declarative Feature Engineering Framework (Not open for public usage) |
| Metaflow | Proprietary | Netflix | Netflix's human friendly Python/R library for Machine Learning. Has robust Feature Engineering and other attributes. (Open for public usage and contribution) |
| Feast | Open-source | Feast-dev, Tecton | Popular open-source Feature Store. Very complete and competent data platform with Python, Spark and Redis. Integrates with many systems and is very customizable. Can be set up with Kubernetes. |
| Hopsworks | Open-source | LogicalClocks | Open-source Feature Store. Used by Amazon Sagemaker. Very hardware demanding. |
| Butterfree | Open-source | QuintoAndar | Open-source tool used for building Feature Stores using Python and Spark. |

## CI/CD for Machine Learning
### Continuous Integration 
A way that when code and data is updated it can be reproducible across all teams and projects.
CI workflows examples:
- Running a versioning the training and evaluation for every commit to the repository
- Running and comparing experiments runs for each Pull Requests to a certain branch
- Trigger a new run periodically
### Continuous Deployment
A method to automate the deployment of the new release to production or any environment 
CD workflow examples:
- Verify the requirements on the infrastructure environment before deploying it
- Test the model output based on a known input
- Load testing and model latency
### Popular Tools
| Tools | License | Developer | Observations |
|---|---|---|---|
| CML (Continuous Machine Learning) | Open-source | Iterative | Most popular tools for CI/CD specific for Machine Learning. By the same developers of DVC, it can be integrated into it. Can be easily used with Github Actions or Gitlab CI/CD. |
| Jenkins | Open-source | Jenkins CI | Jenkins is a popular tool for regular CI/CD that can be used for Machine Learning after some configuration. It is a popular choice among some MLOps projects that intend to run tests on local hardware or heavily configured cloud services. |

## Continuous Monitoring
When models are deployed, it needs to be monitored to assure performance. 
### Things to Monitor
1. Performance
    - Evaluate model performance based on metrics and logging its decision and outcomes
2. Data Issues and Threats
    - Models are increasingly driven by complex feature and automated workflows. Its not unusual for data inconsistencies and errors to reduce model performance - models are susceptible to attacks like injection of data
3. Explainability
    - Explain model decisions is vital for improvement but also for accountability reason
4. Bias
    - Ability to detect and mitigate bias during development process is difficult but necessary
5. Drift
    - The statistical properties of the target variable, which the model is trying to predict changes over time in unforeseen ways. This causes problems because the predictions becomes less accurate as time passes, producing concept drift

### Popular Tools
| Tools | License | Developer | Observations |
|---|---|---|---|
| IBM Watson OpenScale | Proprietary | IBM | Monitors models deployed to IBM Watson Machine Learning. Monitors fairness, explainability and drift. Has tools for managing and correcting problems or inaccuracies in production. |
| OpenShift | Open-source | Red Hat | Kubernetes based system able to deploy various types of applications. It is platform agnostic and can be used for any type of application. Can be useful when a model is heavily integrated into a microservice environment. |
| Seldon Core | Open-source | SeldonIO | Deploys models into microservices with Advanced Metrics, Request Logging, Explainers, Outlier Detectors, A/B Tests, Canaries and more. Platform agnostic and works with many Machine Learning frameworks |
| Tensorflow Extended (TFX) | Open-source | Tensorflow | Deploy Tensorflow Models as API and has monitoring capabilities. |
