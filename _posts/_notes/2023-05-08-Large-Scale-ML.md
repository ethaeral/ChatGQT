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
### Data Processing
### Processing Orchestration

## Exploration
### Workspaces

## Experimentation
### Frequentist AB Testing
### Bayesian AB Testing
### Multi-Armed Bandit
### Impact Estimation

## Large-Scale Training
### Basic Models
### Deep Learning Models
### Model Validation

## Productionization
### Productionization

## Hosting
### Data Hosting
### Model Hosting