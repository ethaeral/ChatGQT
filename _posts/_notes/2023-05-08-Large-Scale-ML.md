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
Producer        -> Broker Cluster -> Consumer (Storage)
binlogs 
segment idx
change streams
#### Apache Kafka
An open-source software platform which provides a way to handle real time data streaming
(Inside of Apache Kafka)
Producers (Clickstream logs) -> Broker Cluster with Leader Election and Replication (Zoo Keeper) -> Consumer (Storage)
#### Amazon Kinesis
An AWS product that provides a way to handle real-time data streaming
Producers (Clickstream logs or any streaming data for ingestion) -> Shards -> Consumers(Storage)
#### Zookeeper
A service designed to reliably coordinate distributed systems via naming service, configuration management, data synchronization, leader election, message queuing, or notification systems
#### Database
A tool used to collect and organize data. Typically, database management systems allow users to interact with the database
#### OLTP
Online transaction processing. A system that handles (near) real-time business processes. For example, a database that maintains a table of the users subscribed to Netflix and which is then used to enable successful log-ins would be considered OLTP. This is opposed to OLAP.
#### OLAP
Online analytical processing. A system that handles the analytical processes of business, including reporting, auditing, and business intelligence. For example, this may be a Hadoop cluster which maintains user subscription history for Netflix.

Producer (OLTP) -> Broker Cluster -> Consumer (OLAP)

#### Availability Zone
Typically a single data center within a region which has two or more data centers. The term multi-AZ implies that an application or software resource is present across more than on AZ within region. This strategy allows the software resources

#### Live Videos / Live Streams
- Ingesting video content from traffic cameras, security cameras, video streaming services
- Example: HTTP Live Stream HLS
    - H.264 compression
    - AAC for sound
    - Effectively chops MP4s up into segments send them out over HTTP

Producers -> Collector Layer -> Broker Cluster -> Consumer (Storage)
- Cameras

#### Batch Ingestion
- Databases
    - Periodic 'snapshots' of the databases
    - Useful when onboarding a new database to be ingested
Producer              -> Broker Cluster -> Consumer
- mySQL mysqldump
- Cassandra CQL copy
- MongoDB mongoexpert

#### Ingestion Consideration
- Size of individual data
- Rate at which data comes in
- Support of data types (changing data types)
- High-availability(multi-AZ) and fault tolerance
### Data Storage
#### Hard Disk Drive
A storage device which operates by settings bits on a spinning magnetic disk. THe capacity and the read/write performance of the HDD are the main characteristics to consider when using an HDD within a particular system
#### Data Replication
A strategy used to mitigate the potential data loss in the event of a system or component failure. IIN the most basic form, it involves writing identical data to more than one device or location. MOre efficient techniques like erasure coding incorporate mathematics to recover lost data without referring to an explicit copy of the data.
#### Hadoop Distributed File System
An open-source Apache software product which provides a distributed storage framework
               -> Name Node Cluster with hot stand by -  Keeps track of which bit of data goes where
Request -> HDFS Client -> Data Node 2
                -> Data Node 1

Uses erasure coding instead of simple replication with an XOR to create a parity bit or something more advance like Reed-Soloman encoding

Sometimes data replication can happen through the process of the producer, broker, and consumer acknowledgements to prevent that we can use Kafka Connector sink -> Exactly-once Semantics

Connects broker and consumer, will guarantees no duplicates
#### Avro
A row-oriented data serializer provided by Apache
- Good for queries which need all columns
- Good for heavy write load
- JSON schema supports evolutions
#### Parquet
A column-oriented data storage format provided Apache
- Good for queries which need some columns
- Good for heavy read load
- Schema challenged with evolution
- Good for sparse data
- Good compression performance
#### Exactly-once Semantics 
Guarantees that an object within a distributed system is processed exactly once. Other semantics include maybe, at-least-once and at-most-once
#### Kafka Transactions
- Uses unique transactional IDs for each producer tied to metadata to ensure that each data committed to the broker is complete -> Sometimes Kafka Streams API simplifies code
### Data Processing

Given the example where we want to predict whether a user will cancel their subscription

Where the data pipeline would look like
Producers(Clickstream logs) -> Broker Cluster (Kafka) -> Consumer (HDFS Data Node)

Message Processing would include
- Aggregation
    - counting how many searchers per user
- Join
    - Combining messages from separate click stream events
- Transformations
    - Nth month of the year instead of the date

Requirements for this structure
- Cluster resource management (CPU, RAM)
- Computational dependency management (locality)
- Manage saving final results to HDFS
- Bonus: Share same HDFS cluster

For this example we'll use Apache Spark and Apache Yarn installed on HDFS cluster

The interaction between Spark and Yarn is that

YARN AM ->  Spark Driver
YARN containers <- Spark Executors

Resource manger -> Node manager: AM -> Driver, Container-> Executor
  |-> Node manager: Container-> Executor

With this the single point of failure would be resource manager, so we could add zookeeper with an active/passive hot-standby


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
A software interface which provides implicit parallelism and fault tolerance across a cluster of machines - Gives users ease of access to run code across cluster 
- Driver
    - Converts user's code to a set of tasks (unit of work)
    - Schedules tasks across executors
- Cluster Manager
    - YARN schedules jobs submitted to cluster
- Executor 
    - Runs tasks
    - Returns results to drivers

Action Flow:
Driver will receive a job
Driver will
- Verifies schema, types, data structures
- Optimizes your code, with an 'optimized logical plan'
    - Reduce code to RDD DAG (Resilient Distributed Dataset Direct Acyclic Graph)
    - DAG Scheduler (stages)
    - Task scheduler
- Form physical plan
    - Creates tasks and DAG schedulers, will then execute this plan with task schedulers
    - Executors will return results to Driver

#### Apache YARN
A software product responsible for managing compute resources across a cluster of machines
- A resource negotiator that 
    - Scheduler allocates cluster resources
    - Applications manager accepts jobs to be run on cluster
- Node Manager (Per-node)
    - Negotiates with Resource manager for resources requested by Application Master AM
- Application Master
    - Reports resource usage to resource manager
    - Negotiates with the scheduler for containers
- Containers
    - Abstraction representing RAM CPU DISK
Action flow:
Client sends a task to resource manager
Resource manager the starts a container an application master on the node manager
Application master can ask for more resources from the resource manager
Resource manager will tell the AM where free nodes resources are and starts those containers 
Containers will run code
To check in on the processes, you get status from AM
After its done the AM will deregister with the RM 
RM will clean up the resources that were deregistered
#### Elastic MapReduce
An Amazon Web Services product which provides users access to a Hadoop cluster
#### Jupyter Notebook
A project jupyter product which provides an interactive workspace to execute code.
### Processing Orchestration
In an example where we want to get fresh data every 24hrs
- Get interaction data from the search page and the carousel
- Merge interaction data into one dataframe
- Join interaction features with labels
- Ideally not to do this manually
- Handle data dependencies (serial, parallel)
- Manage potentially thousands of scheduled jobs
- We can here use Apache Airflow
#### Apache Airflow
A workflow management system that provides users a way to author, schedule and execute software
- DAG Directed Acyclic graph
get_data_A -\
        join_AB ----\
get_data_B -/       join_ABC
        get_data_C -/
Airflow offers
- Webserver (Multiple instances behind load balancers)
    - Flask app allows users to trigger DAGs
    - Browse DAG history (stored in database)
- Scheduler - Can be set with active/passive hot standby and support concurrency without can be single point of failure - also reduces duplications
    - Monitors database to check task states
    - Fetch DAGs from DAG store (S3 DFS)
    - Send DAG tasks to execution queue (Could be RabbitMQ which can be scaled for availability)
    - Writes DAG runs to database (Will have to active/passive hot standby as well) for history
- Worker 
    - Pulls the task queue
    - Runs the tasks
    - Stores task state to the database
    - Implemented through celery worker
        - If one of worker goes down, celery will reassign the work

Action flow:
User will trigger the DAG in the webserver
Webserver fetc DAG from DAG store like S3
Webserver will schedule the DAG with scheduler
Scheduler enqueue Tasks in parallel to the Queue
Works will constantly pull tasks to work on and write to database when complete
Scheduler will periodically check if tasks are done
This allows schedulers to then queue more jobs that were dependent on other jobs to get done
When all tasks ae done then scheduler will add DAG History to the Database
This allows webserver to getDAGStatus
And webserver will relay to client DAG status
- Check stats of DAG Run
- Failures
- Runtime
- Can alter schedule

## Exploration
### Workspaces
- Explore features, labels, and models
    - Completeness of data
        - Stability
        - Availability
        - Free of bias
            - Positive feedback loop
    - Pre-trained models
        - Transfer learning
            - Embedding layers
    - Explainability
        - Shapley values
        - Lime
        - Deeplift
    - Model types
        - Layered, ensemble, auto-ml
    - Feature importance
- Leverage team resources
    - Team packages
    - Collaboration
- Environment management
    - Individualized exploration
    - Production-ready for serving predictions
- Spark/HDFS access
- Asynchronous support
    - Training, hyperparameter tuning, evaluation
- Data access governance
    - Protected data

Tools:
- Jupyter hub / notebooks
    - Amazon sagemaker studio
    - Google collab
    - Azure ML workspace


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