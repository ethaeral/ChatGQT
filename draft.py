def two_number_sum_CPU_constraint(array: list[int], target_sum:int)-> list[int]:
    """Takes an array and sees if there are two values that adds up to a given sum. 
    
    Strategy:
        Uses a hash map to keep track of numbers that is needed in the next loops to add the current loop index value to the target sum. When looping into the array, checking if it is a sum pair, the operation will be one. It will take one pass through the array. This will give a faster computation at the expense of memory

    Args:
        array (list[int]): 
            An array that will be traversed to find two integers to be summed to target sum
        target_sum (int): 
            An number given to be a target for two pairs of number to sum up to in a given array
    
    Returns:
        list[int]: 
            Either an empty array or two values from inputted array that adds up to the target_sum
        
    Constraints:
        array: len(array) < sys.maxsize & len(array) >= 0 
        target_sum: target_sum > -2.15B & target_sum
        return: len(return) == 2
        CPU constraint
    """
    
    # Prevents initializing variables when array is empty 
    if len(array) == 0: return []
    # Serves as a tracker for what numbers we are looking for in respect to what we have evaluated already
    hash_map: dict[int,int] = {}
    for num in array:
        if hash_map.get(num) != None:
            # Returns first two numbers whose sum is targetSum
            return [hash_map[num], num] 
        else:
            # Creates entry where value is the current number in the loop and the key is the number that would need to be in array for the summation be the targeted number
            hash_map[target_sum-num] = num
    return []


print(two_number_sum_CPU_constraint([23,2,7],9))

def two_number_sum_memory_constraint(array: list[int], target_sum:int)-> list[int]:
    """Takes an array and sees if there are two values that adds up to a given sum. 
    
    Strategy:
        Two pointers

    Args:
        array (list[int]): 
            An array that will be traversed to find two integers to be summed to target sum
        target_sum (int): 
            An number given to be a target for two pairs of number to sum up to in a given array
        
    Returns:
        list[int]: 
            Either an empty array or two values from inputted array that adds up to the target_sum
        
    Constraints:
        array: len(array) < sys.maxsize & len(array) >= 0 
        target_sum: target_sum > -2.15B & target_sum
        return: len(return) == 2
        memory constraint
    """
    
    # Prevents initializing variables when array is empty 
    if len(array) == 0: return []
    # Serves as a tracker for what numbers we are looking for in respect to what we have evaluated already
    hash_map: dict[int,int] = {}
    for num in array:
        if hash_map.get(num) != None:
            # Returns first two numbers whose sum is targetSum
            return [hash_map[num], num] 
        else:
            # Creates entry where value is the current number in the loop and the key is the number that would need to be in array for the summation be the targeted number
            hash_map[target_sum-num] = num
    return []

"""
Think about what is the algo going to used for
    - Restrictions: Memory, Hardware

Think of possible solutions for ea. scenario
- Memory constraint -> just use more CPU
    - Streaming
- CPU constraint -> just use more Memory
    - Per CPU core / thread 
- Time constraints -> Batch / Chunking / Parallel processing 

"""