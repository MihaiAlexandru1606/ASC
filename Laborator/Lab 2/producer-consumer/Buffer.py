from threading import Semaphore


class Buffer(object):

    def __init__(self, buffer_size):
        self.buffer_size = buffer_size
        self.buffer = []
        self.lock_consumer = Semaphore(0)
        self.lock_producer = Semaphore(buffer_size)

    def put(self, item):
        self.lock_producer.acquire()
        self.buffer.append(item)
        self.lock_consumer.release()

    def get(self):
        self.lock_consumer.acquire()
        item = self.buffer.pop()
        self.lock_producer.release()

        return item

    def size(self):
        return self.buffer_size