import { LinkedList } from "./LinkedList";

describe('testing LinkedList class', () => {
    let data = 'test123'
    let data2 = 'test321'
    let data3 = 'test3211'
    let data4 = 'testdata'
    let newSLL = new LinkedList(data)


    test('creating new linked list returns length of 1', () => {
        expect(newSLL.len).toBe(1);
      });
  
      test('linked list stores data correctly', () => {
          expect(newSLL.head.value).toBe(data);
        });

    test('set functions creates a new node, if not given an id and returns length of 2', () => {
        newSLL.set(data2)
        expect(newSLL.len).toBe(2);
    });

    test('set functions changes node value, if given and id and returns length of 2', () => {
        let secondNodeId = newSLL.head.next.id
        newSLL.set(data3, secondNodeId)
        expect(newSLL.len).toBe(2);
        expect(newSLL.head.next.value).toBe(data3)
    });

    test('get functions finds node by id', () => {
        let secondNodeId = newSLL.head.next.id
        let foundNode = newSLL.get(secondNodeId)
        expect(foundNode.value).toBe(data3)
    });

    test('del functions del node by id', () => {
        newSLL.set(data4)
        let secondNodeId = newSLL.head.next.id
        newSLL.del(secondNodeId)
        expect(newSLL.head.next.value).toBe(data4)
    });

  });