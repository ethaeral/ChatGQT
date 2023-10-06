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
- Compute hase code from key
- Compute index from hash code
- Search linked list

Worst Case O(N)
Look up time O(1)

2. Binary Search Tree For Look Up
Implement a look up system using a balanced binary search tree. The advantage of this is potentially using less space, since we no longer allocate a large array.
Worst Case O(N)
Look Up Time O(log N)