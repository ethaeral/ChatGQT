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

## Optimization Techniques
1. BUD 
    - Bottlenecks
        - Part of algo that slows down overall runtime
            - One time work that slows down
            - Chuck of work that is down repeatedly
    - Unnecessary Work
        - Breaking out of loops early
    - Duplicated Work
        - Create hashing to for the same work
## Solve Techniques
1. DIY
    - Think about if you were gonna solve the problem in real life
2. Simplify and Generalize
    - Simplify or tweak some constraint such as a data type
    - Solve for that and then adapt it to the more complex version
3. Base case and build
    - Solve the problem for the base case then build to more special cases
4. Data Structure Brainstorm
    - Just list all data structures and list pros and cons

## Customized Flow
Pre Dev Cycle:
- Break down problem todos, constraints, unique information, problem type?
- How would you do this if this was a real world situation?

Intial Pass (Brute Force Algortihm)
- Test **
    - General: input big enough, not a special case
    - Special: Null, single ele, extreme cases, etc
- Walk
    - Deconstruct solution and explain step by step
    - State time and space complexity
- Run (Code)
    - Modularized
    - Error checked
    - Classes and structs when appropriate
    - Good variables
    - Make comments of places where it needs refractoring
- Bud
    - Bottleneck, unnecessary work, duplicated work
- Bloom (Optimize) **
    - Check if all info has been used
    - Data structure brainstorm with pros and cons
    - State time opt and space opt solution
    - State BCR

Next Pass (Optimized Algorithm)
- Test **
    - More general input tests
    - Conceptual tests
    - Hot spots
        - Base cases
        - Interger division
        - Null node in binary tree
        - Start and end of iterations
- Walk
    - Deconstruct solution and explain step by step
    - State time and space complexity
- Run (Code)
    - Modularized
    - Error checked
    - Classes and structs when appropriate
    - Good variables
    - Make comments of places where it needs refractoring
- Bud 
    - Bottleneck, unnecessary work, duplicated work
- Bloom (Optimize) **
    - Check if all info has been used
    - Check if there's more that can be optimized

Post Dev Cycle:
- Further testing 
    - Look for weird spaces
    - Hot spots

