import {v4 as UUIDV4} from 'uuid';

export class DDLNode {
    id: string
    value: any;
    prev: DDLNode | null;
    next: DDLNode | null;
    constructor(data: any){
        this.id = UUIDV4();
        this.value = data;
        this.prev = null;
        this.next = null;
    }
}

export class LinkedList {
    head: DDLNode
    len: number

    constructor(data:any){
        this.head = new DDLNode(data);
        this.len = 1;
    }

    public set(data: any, id?: string){
        if (id){
            let node: DDLNode = this.traverse('id', id)
            node.value = data;
            return {node, len:this.len}
        }else{
            let node: DDLNode = this.traverse('next', null)
            let current = node.next = new DDLNode(data)
            current.prev = node
            this.len += 1
            return {node: current, len:this.len}
        }
    }

    public get(id: string){
        return this.traverse("id", id)
        
    }

    public del(id: string){
        let foundNode = this.get(id)
        let previousNode = foundNode.prev
        let nextNode = foundNode.next
        previousNode.next = nextNode
        nextNode.prev = previousNode
        this.len -= 1
        return foundNode.id
    }

    private traverse (attr: string, stop: string | null){
        let current: DDLNode = this.head;
        if (current !== null){
            while (this.getAttr(attr, current) !== stop){
                current = current.next   

            }
            return current
        }else{
            return null;
        }
  
    }   

    private getAttr (attr: string, node: DDLNode){
        switch(attr){
            case "id":
                return node.id;
            case "value": 
                return node.value;
            case "next":
                return node.next
            default:
                return null      
        }
        return null
    }


}