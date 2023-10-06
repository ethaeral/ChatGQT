---
title: Code Solving Flow
date: 2023-10-03 00:00:00 -0500
categories: [Algorithms]
tags: [forme, interview prep, notes]
---

A Problem Solving Flow
1. [Listen](#listen)
    - Consider all problem information for an optimal algorithm
2. [Example](#example)
    - Check examples if they are a special case or if they are too small
3. [Brute Force](#brute-force)
    - State naive algorithm with runtime and list possible optimization
4. [Optimize](#optimize)
    - Work through BUD (Bottlenecks, Unnecessary Work, Duplicated Work) optimization
    - Solve manually, reverser engineer it, how did you solve it
    - Make a time vs space tradeoff
5. [Walk Through](#walk-through)
    - Walk through your approach in detail before coding - white boarding
6. [Implement](#implement)
    - Psuedo code -> Beautiful Modular Code
7. [Test](#test)
    - Unusual or nonstandard code
    - Hot spots 
    - Small, special, edge test cases

## Listen
- Make sure you hear the problem correctly
- Ask questions about what you're unsure about
- Record any unique information in the problem
## Example
- Specific, it should use real numbers or strings
- Sufficiently large
- Not a special case
## Brute Force
- State its a brute force even if it's terrible
## Optimize
- Look for unused info
- Use fresh examples
- Make time vs space trade offs
- Pre compute info
- Use hashtables
- Think about best conceivable runtime
## Walk Through
- Make sure to have a step by step in plain english what will be happening in your code - break it down into small steps
## Implement
- Code for far top left of the board - avoid slanted lines
- Write beautiful code
    - Modularized code
    - Error checks
    - Use classes and structs when appropriate
    - Good variable names
    - If you see something you can refractor you can comment on it
## Test
- Conceptual test, can you explain your code
- Check weird looking code
- Check hot spots
    - Base cases
    - Interger division
    - Null nodes in binary tree
    - Start and end of iterations through linked list
- Small test cases
- Special test cases
    - Null
    - Single element values
    - Extreme cases
    - Other special case

1. Break down question and note unique information
2. Create intial test cases
    - General
        - Input sufficently large
        - Not a special case
    - Special test cases
        - Null
        - Single element values
        - Extreme cases
        - Other specialcases
3. Walk through brute force algo
    - State the time and space complexity
4. Optimize
    - Check for unused info
    - Make time vs space trade offs
        - State best solution for time opt and space opt
            - State pros and cons to each 
    - State best concievable run time
    - Common optimizations:
    - Pre compute info -> hashtables
5. Walk through optimized algo
    - Make sure to have a step by step plain english what will happen
6. Implement
    - Write beautiful code
        - Modularized
        - Error checked
        - Use classes and structs when appropriate
        - Good variable names
        - If you need to refractor use comments
7. Create more tests
    - More general input tests
    - Conceptual tests -> explain your code
    - Look for weird spaces
    - Hot spots
        - Base cases
        - Interger division
        - Null node in binary tree
        - Start and end of interations through linked list


