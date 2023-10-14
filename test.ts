/*
types: describes shapes of data
interfaces: declaration merging
enum: key-value pair - predefined data
*/

/*
Hash Table

Common hash table implementation

Added implementation by:
Hashing
- Integer Universe Assumption
    - Division
    - Multiplication
    - Universal
    - Dynamic perfect
    - Static Perfect
Collision Resolution
- Seperate chaining
    - Linked list
    - Self balancing binary search tree
    - Dynamic perfect hashing
- Open addressing
    - Linear probing
    - Quadratic Probing
    - Double hashing
- Other collision resolution strategies
    - Coalesced hashing
    - Cuckoo hashing
    - Hopscoth hashing
    - Robin Hood hashing
Resizing
- All at once 
- 50% new allocation
- Linear hashing

*/

type HashTableData = {
    linkedList?: any;
    binarySearchTree: any;
    dynamic: any;
    linear: any;
    quadratic: any;
    double: any;
    coalesced: any;
    cuckoo: any;
    hopscoth: any;
    robinHood: any;
}

type HashTableConfiguration = {
    collision: string;
    hashing: Function;
    resizing: string;
}

class HashTable {
    strategy: HashTableConfiguration;
    data: HashTableData;
    hashing;
    resizing;

    constructor(config: HashTableConfiguration){
        this.strategy = config;
    }

    // take configuration and set up new hashtable type
    initialize(){}

    create(){
        // switch case for functions for each strat
    }
    
    read(){
        // switch case for functions for each strat
    }
    
    update(){
        // switch case for functions for each strat
    }
    
    delete(){
        // switch case for functions for each strat
    }

    resize(){
        // switch case for functions for each strat
    }

    hash(){
        // switch case for functions for each strat
    }
}