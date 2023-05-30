---
title: System Design Fundamentals
date: 2023-05-25 00:00:00 -0500
categories: [Core]
tags: [forme, interview prep, notes]
---
Context Collection:     
What system are you actually building
What functionality system will have to support
What characteristic are we going to value in our system

Justification
Justify your solution
Rationalize certain choices

Design Fundamentals
4 Broad Categories
1. Underlying foundational knowledge
2. Key characteristics of Systems
3. Actual Components of a System
4. Tech

### Underlying Foundational Knowledge
Client Server Models / Architecture 
Client
A machine or process that requests data or service from a server
Server 
A machine or process that provide data or service for a client usually by listening for incoming network calls
Client-Server Model
The paradigm by which modern systems are designed, which consists of clients requesting data or service from servers and servers providing data or service to client
IP Address
An address given to each machine connected to the public internet. a.b.c.d where all slots can be numbers 0-255
Port
In order for multiple programs to listen for new network connections on the same connections on the same machine without colliding, they pick a port to listen on, A port can be represented between 0-65,535
Predefined ports
22: Secure shell
53: DNS Lookup
80: HTTP
443: HTTPS
DNS
Short for Domain Name System, it describes the entities and protocols involved in the translation from domain names to IP addresses. Typically machines make a DNS query to a well known entity which is responsible for returning the IP address of the requested domain name in the response.

Network Protocols
