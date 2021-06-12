from queue import PriorityQueue

q = PriorityQueue()

q.put((3, '?'))
q.put((1, 'cat'))
q.put((1, 'dog'))
q.put((2, 'house'))

print(q.queue)

while not q.empty():
    next_item = q.get()
    print(next_item)