---
title: Arrays and Strings
date: 2023-10-06 00:00:00 -0500
categories: [Data Structure]
tags: [forme, interview prep, notes]
---
## Hash Tables
A structure that maps keys to values for an efficient lookup.

### Implementations
1. Array of linked list with hashed code indexes
Insert:
- Compute the key's hash code, which is usually an int or long
- Map the hash code to an index in the array 
    - Usually looks like:  hash(key)%array_length
- At this index, a linked list of key and values are stored. Linked lists are used to prevent collision and lost data

Retrieve:
- Compute hash code from key
- Compute index from hash code
- Search linked list

2. Binary Search Tree For Look Up
Implement a look up system using a balanced binary search tree. The advantage of this is potentially using less space, since we no longer allocate a large array.
Worst Case O(N)
Look Up Time O(log N)

#### Collision Gaurd and Resolution
- Hash implementations:
    - Integer Universe Assumption
        - Hashing by division (mod), multiplication, universal, dynamic perfect, static perfect
        - Division is the most common
    - Hash values should have a uniform distribution
    - Reduce clustering -> k-independent hashing
- Collision Resolution:
    - Separate Chaining (common)
        - Collided items are chained together by
            - Linked list
                - Search $O(n)$
                - Insert $O(1)$
                - Dynamic array may be more cache friendly because the reduction of scattered memory storage
            - Self balancing binary search tree 
                - Search $O(log n)$
                - Insert $O(n log n)$
            - Dynamic Perfect hashing
                - Two level hash table to reduce look up complexity to $O(1)$
                - Buckets of $k$ entries are organized as perfect hash tables with $k^{2}$
                - This is more perfomant when compared the standardlinked list under heavy load
                - Fusion tree for each bucket also results in constant time
    - Open Addressing
        - Every entry record is stored in a bucket array itself and hash resoution is performed through probing. Probing sequences include:
            - Linear Probing
            - Quadratic Probing
            - Double hashing
        - Performance of open address maybe slower when load factor approaches 1
        - Based on open addressing
            - Coalesced hashing
            - Cuckoo hashing
            - Hopscoth hashing
            - Robin Hood hashing


#### Resizing
- Moving all entries to a newly allocated hash table 
- Typically at 50% willstart a new allocation and rehashing is done incrementally through extending prior memory block allocated
- Linear hashing enables dynamic growth or shrinks of the table one bucket at a time

### Time Complexity

| Algorithm | Average | Worst  |
|-----------|---------|--------|
| Space     | $O(n)$  | $O(n)$ |
| Search    | $O(n)$  | $O(n)$ |
| Insert    | $O(1)$  | $O(n)$ |
| Delete    | $O(1)$  | $O(n)$ |

### Space Time Trade Off
Time Optimized: Entire key can be used directly as an index
Memory Optimized: Values can be stored without regard for their keys and binary search or linear search can be used for retrieval 

### Performance
Hash tables turn out to be on average more efficient than search trees or other table look up structures


### Use Cases
- Database indexing
- Caches
- Set

### Terms
- Associative Array
    - Stores a set of (key,value) pairs
- Load Factor
    - $n/m$ 
        - $n$ is the number of entries occupied in the hash table
        - $m$ is the number of buckets
    - Performancce of the hash deteriorates in relation to load factor. Acceptable figures of a load factor should be 0.6-0.75

### Naive Approach

## ArrayList & Resizable Arrays
Resizable array implementation is set to where when the array is full to increase the size by 50% or another value. Access takes O(1) and resizing takes O(n) worst case