Network Protocols

IP
Stands for internet protocol. This network protocol outlines how almost all machine to machine communication should happen in the world. Other protocols like TCP UDP and HTTP are built on top of IP
IPV4 widely used
IPV6 newer not as widely used

TCP 
Network protocol built on top of he Internet Protocol IP. Allows for ordered reliable data delivery between machines over the public internet by creating a connection -> which fixes the IP packet problem -> gives ability to send a lot of data

TCP is usually implemented in the kernel, which exposes sockets to applications that they can use to stream data through an open connection


HTTP
The hypertext transfer protocol is very common network protocol implemented on top of the TCP. Clients make HTTP requests, and servers responds with a response.

Built on top of IP Packet and TCP to build a higher abstraction and gives ability to throw in business logic 

Requests looks like
```
host: str
port: int
method: [key:str]:str
body: bytes
```
Responses typically look like
```
status code: int
headers: [key:str]: str
body: bytes
```

IP Packet
Sometimes more broadly referred to as just a network packet, an IP packet is effectively the smallest unit used to describe data being sent over IP, aside from bytes.
- an IP header which contains the source and destination IP addresses as well as other information related to the network 20-60 bytes
- a payload which is just the data being sent over the network 2^16 bytes 6.5MB

Storage

Databases 
Databases are programs that either use disk or memory to do 2 core things: record data and query data. In general, they are themselves servers that are long lived and interact with the rest of your application through network calls, with protocols on top of TCP or even HTTP

Some databases only keep records in memory, and the users of such databases are aware of the fact that those records may be lost forever if the machine or process dies.

For the most part though, databases need persistence of those records, and thus cannot use memory. This means that you have to write your data to disk. Anything written to disk will remain through power loss or network partitions, so that what is used to keep permanent records.

Since machines die often in a large scale system, special disk partitions or volumes are used by the database processes, and those volumes can get recovered even if the machine were to go down permanently.

Disk
Usually refers to either HDD or SSD. Data written to disk will persist through power failures and general machine crashes. Disk is also referred to non-volatile storage.

SSD is far faster than HDD but also far more expensive from a financial point of view. Because of that HDD will be typically used for data that's rarely accessed or updated

Memory
Short for RAM, data stored in memory will be lost when process has written that data dies

Persistent Storage 
Usually refers to disk, but in general it is any form of storage that persists if the process in charge of managing it dies

#### Latency and Throughput

Latency   
The time it takes for a certain operation to complete in a system. Most often this measure is a time duration, like milliseconds or seconds. You should know these orders of magnitude:
- Reading 1 MB from RAM 0.25 ms
- Reading 1 MB from SSD 1 ms
- Transfer 1 MB over network 10 ms
- Reading 1 MB from HDD 20 ms
- Inter-continental round trip <1 mb 150ms

Throughput   

### Availability
Availability: Fault tolerant 

Availability   
The odds of a particular server or service being up and running at any point in time, usually measured in percentages. A server that has 99% availability will be operational 99% of the time this would be described as having two nines of availability   

High Availability
Used to describe systems that have particularly high levels of availability, typically 5 nines or more; sometimes abbreviated HA

Nines   
Typically refers to percentages of uptime. For example, 5 nines of availability means an up time of 99.999% of the time Below are the downtimes expected per year depending on those 9s:
```
99%: 87.7 hours
99.9%: 8.8 hrs
99.99: 52.6 hrs
99.999: 5.3 mins
```

Redundancy
The process of replicating the parts of a system in an effort to make it more reliable   

SLA  
Short for service-level agreement an SLA is a collection of guarantees given to a customer by a service provider. SLAs typically make guarantees on a systems availability, amongst other things. SLAs are made up of one or multiple SLOs.

SLO
Short for service level objective, an SLO is a guarantee given to a customer by a service provider. SLOs typically make guarantees on a system's availability, amongst other things. SLOs constitute an SLA

SLOs are components of SLAs

Want to avoid single point of failure
Can solve with load balancers and redundancy

Passive redundancy

### Caching
Cache 
A piece of hardware or software that stores data, typically meant to retrieve that data faster than otherwise.

Caches are often used to store responses to network requests as well as results of computationally-long  operations

NOte that data in a cache can become stale if the main source of truth for that data (the main database) gets updated and the cache doesn't

Cache hit
When requested data is found in a cache

Cache Miss
When requested data could have been found in a cache but isn't. THis is typically used to refer to a negative consequences of a system failure or poor design choice

Cache Eviction Policy
The policy by which values get evicted or removed from a cache. Popular cache eviction policies include LRU, FIFO, LFU

Content Delivery Network
A CDN is a third party service that acts like a cache for your servers. SOmetimes, web applications can be slow for users ina  particular region if your servers are located only in another region. A CDN has servers all around the world, meaning that the latency to a CDN's servers will almost always be far better than the latency to your servers. A CDN's servers are often referred to as PoPs points of presence. Two of the most popular CDNs are Cloudflare and Google Cloud CDN


Caching is good for 
Static or immutable data
If you only have one thing reading and writing data
If don't care about staleness, consistency, accuracy
If you have a system to properly invalidate stale data, LRU, FIFO, LFU   

### Proxies
Forward Proxy  
A server that sits between a client and servers and acts on behalf of the client, typically used to mask the client's identity. Note that forward proxies are often referred to as just proxies  

Reverse Proxy  
A server that sits between clients and servers and acts on behalf of the servers, typically used for logging, load balancing, or caching.  

Nginx   
Pronounced "engine X" - not "n jinx" nginx is very popular web server that's often used as a reverse proxy and load balancers

### Load Balancers
Load Balancer  
A type of reverse proxy that distributes traffic across servers. Load balancers can be found in many parts of a system, from the DNS layer all the way to the database layer.   

Software and hardware load balancers

Server-Selection Strategy  
How a load balancer chooses servers when distributing traffic amongst multiple servers. Commonly used strategies include round-robin, random selection, performance-based selection (choosing the server with best performance metrics, like the fastest response time or the least amount of traffic), and IP-based routing.  

DNS Round robin approach
Goes through all the servers and then back to the top 

Weighted round robin, still do the same but you can direct more computations to one of the servers

IP, will hash and route to the server

Hot Spot
When distributing a workload across a set of servers , that workload might be spread unevenly. THis can happen if your sharding key or your hashing functions are suboptimal, or if your workload is naturally skewed: some servers will receive a lot more traffic than others, thus creating a hot spot

### Hashing

If you have a system that relies heavily on caches,m hashing might be better suited to match server to request type - by modding the hash by the number of servers. This assigns association from client to server, but if servers are down ->  caching does not work because it will become a mess

Consistent Hashing  
A type of hashing that minimizes the number of keys that need to be remapped when a hash table gets resized. It's often used by load balancers to distribute traffic to servers; it minimizes the number of request that get forwarded to different servers when new servers are added or when existing servers are brought down.   

Imagine a circle and its has number value as it traversed, the servers will be plotted by configuration ont he circle. Each client will be assigned place on the circle by hashing - to find which client is to which server, it will follow in a clockwise path until next server is the one server

If server dies then the client just gets moved to the next server 

You can also hash a server or the client onto the circle multiple times 

Rendezvous Hashing   
A type of hashing also coined highest random weight hashing. Allows for minimal re-distribution of mappings when a server goes down.

For every client it will be rank each server, and pick the highest ranking server - and when servers are removed, it goes to the second ranking server


SHA  
Short for  Secure Hash ALgorithms, the SHA is a collection of cryptographic hash functions used in the industry. These days, SHA-3 is popular choice to use in a system.

### Relational Database

