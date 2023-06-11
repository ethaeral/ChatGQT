---
title: Large Scale Machine Learning Notes
date: 2023-04-17 00:00:00 -0500
categories: [Core]
tags: [forme, interview prep, notes]
---
## Data Support
### Data Ingestion
#### Clickstream
An ordered series of interactions that users have with some interface. In the traditional sense, this can be literal clicks of a mouse on a desktop browser. Interactions can also come from touchscreens and conversational user interface
#### Change Data Capture
The process of recording changes in the data within the database system. For instance, if a user cancels their Netflix subscription, then the row can be recorded and referenced later for analysis or audit purposes
#### Apache Kafka
An open-source software platform which provides a way to handle real time data streaming
#### Amazon Kinesis
An AWS product that provides a way to handle real-time data streaming
#### Zookeeper
A service designed to reliably coordinate distributed systems via naming service, configuration management, data synchronization, leader election, message queuing, or notification systems
#### Database
A tool used to collect and organize data. Typically, database management systems allow users to interact with the database
#### OLTP
Online transaction processing. A system that handles (near) real-time business processes. For example, a database that maintains a table of the users subscribed to Netflix and which is then used to enable successful log-ins would be considered OLTP. This is opposed to OLAP.
#### OLAP
Online analytical processing. A system that handles the analytical processes of business, including reporting, auditing, and business intelligence. For example, this may be a Hadoop cluster which maintains user subscription history for Netflix.
#### Availability Zone
Typically a single data center within a region which has two or more data centers. The term multi-AZ implies that an application or software resource is present across more than on AZ within region. This strategy allows the software resources
### Data Storage
#### Hard Disk Drive
A storage device which operates by settings bits on a spinning magnetic disk. THe capacity and the read/write performance of the HDD are the main characteristics to consider when using an HDD within a particular system
#### Data Replication
A strategy used to mitigate the potential data loss in the event of a system or component failure. IIN the most basic form, it involves writing identical data to more than one device or location. MOre efficient techniques like erasure coding incorporate mathematics to recover lost data without referring to an explicit copy of the data.
#### Hadoop Distributed File System
An open-source Apache software product which provides a distributed storage framework
#### Avro
A row-oriented data serializer provided by Apache
#### Parquet
A column-oriented data storage format provided Apache
#### Exactly-once Semantics 
Guarantees that an object within a distributed system is processed exactly once. Other semantics include maybe, at-least-once and at-most-once
### Data Processing
#### Recommendation Carousel
A component within a graphical or conversational user interface which presents recommendations to a user. This can include products, ads, and media content.
####  Central Processing Unit
A general purpose compute resource responsible for executing computer programs.
#### Seasonality
The predictable changes of data throughout the calender year
#### Parallelization
When two or more computer programs are executed at the same instant across more than one processor.
#### Random Access Memory
A device on a computer which stores the data and machine code of a running computer program
#### Apache Spark
A software interface which provides implicit parallelism and fault tolerance across a cluster of machines
#### Apache YARN
A software product responsible for managing compute resources across a cluster of machines
#### Elastic MapReduce
An Amazon Web Services product which provides users access to a Hadoop cluster
#### Jupyter Notebook
A project jupyter product which provides an interactive workspace to execute code.
### Processing Orchestration
#### Apache Airflow
A workflow management system that provides users a way to author, schedule and execute software

## Exploration
### Workspaces
#### Automated Machine Learning
A strategy which automates the process of applying features and labels to a machine learning model
#### Data Governance
The method of managing, using, and protecting an organization's data

## Experimentation
### Frequentist AB Testing
#### A/B Testing
The process of providing two or more different experiences across two or more subgroups of a population. The goal is the measure the change in behavior of the subgroups upon receiving the respective experiences
#### A/A Test
An A/B test in which the experience being tested are identical to one another. THis is done in an effort to determine statistical validity of the A/B tool, the metric being monitored, and the analysis process being used.
#### User Agent 
An identifier used to describe the software application which a user is using to interact with another software application. For instance, an HTTP request to a website typically includes the user agent soo that the website knows how to render the webpage.
#### Session ID
A unique identifier assigned to a user to keep track of a user's connected interactions. For instance, a session may include a user logging in, purchasing an item, and logging out. Here, the session ID would be used to reference the group of those three interactions. This session ID can be stored in the user's internet browser as a cookie
#### Cookie
A small piece of data stored by a browser which indicates stateful indicates stateful information for a particular website. For instance, a cookie can be stored in your browser after you log in to a website to indicate that you are logged in. This will stop subsequent pages from asking you to log in again.
### Bayesian AB Testing
#### Beta Distribution
This distribution is used to model percentages and proportions such as click-through probabilities
### Multi-Armed Bandit
#### Multi-Armed Bandit
A process which provides a number of choices
### Impact Estimation
#### Shadow Test
Running two or more versions of software in parallel while only surfacing the result of one of the experiences to the end user. THis is done in an effort to gauge the difference between the software versions
#### Sample Selection Bias
The bias that occurs when sampling a population into one or more subgroups at random results in a systematic inclusion or exclusion of some data.
#### Experiment Collision
The event where one experiment unintentionally influences the results of the one or more separate experiments

## Large-Scale Training
### Basic Models
#### MLlib
A library provided by APache SPark which provides Spark clusters access to machine learning algorithms and related utilities. MLlib provides a Dataframe-based API which is unofficially referred to as SparkML
### Deep Learning Models
#### Model Parallelism
A machine learning model training strategy used to maximize the utilization of compute resources in which the model is distributed across two or more devices
#### Data Parallelism
A machine learning model training strategy used to maximize the utilization of compute resources in which the model is distributed across two or more devices 
#### Graphic Processing Unit
A specialized device that has many cores, allowing it to perform many operations at a time.
GPUs are often used within deep learning to accelerate training of neural networks by taking advantage of their ability to perform many parallel computations.
#### Concurrency
When two or computer programs share a single processor.
### Model Validation
#### Hyperparameters Optimization
The process of searching for the best possible values of the hyperparameters of some machine learning model
## Productionization
### Productionization
#### Recall
Also sensitivity, is the proportion of true positives which are correctly classified
#### Precision
The number of true positives divided by the true positives plus false positives.

## Hosting
### Data Hosting
#### In-memory Database
A database which relies either solely or primarily on the RAM of a computer
#### Distributed Cache
A cache which distributed across two or more machines
### Model Hosting
#### Numba
A just-in-time Python complier which resolves a subset of the python programming language down to machine code