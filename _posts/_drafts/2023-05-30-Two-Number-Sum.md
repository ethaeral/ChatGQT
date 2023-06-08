---
title: Two Number Sum Problem
date: 2023-05-30 00:00:00 -0500
categories: [Core]
tags: [forme, interview prep, notes]
---

## Two Number Sum

Write a function that takes in a non-empty of distinct integers and an integer representing a target sum. If any two numbers in the input array sum up to the target sum, the function should return them in an array, in any order. If no two numbers sum up to the target sum, the function should return an empty array.

Note that the target sum has to be obtained by summing two different integers in the array; you can't add a single integer to itself in order to obtain the target sum.

You can assume that there will be at most one pair of numbers summing up to the target sum.

Sample input

```
array = [3,4,5,-2,2,33,12]
targetSum = 10
```

Sample output

```
[-2,12] or [12,-2]
```

## Solution for Memory Optimization

```py
def twoSum(nums: List[int], target: int) -> List[int]:
    for i in range(len(nums)):
        for j in range(i + 1, len(nums)):
            if nums[j] == target - nums[i]:
                return [i, j]
```

## Solution for CPU Optimization

```py
def twoSum(nums: List[int], target: int) -> List[int]:
    if len(nums) == 0: return []
    hash_map: dict[int,int] = {}
    for num in nums:
        if hash_map.get(num) != None:
            paired_values = [hash_map[num], num]
            return paired_values
        else:
            hash_map[target_sum-num] = num
    return []
```

## Solution for Speed Optimization

Split nums into even amounts of chunks
Run code concurrently 

```py
def twoSum(nums: List[int], target: int, chunk_speed: int) -> List[int]:
    pass
```
