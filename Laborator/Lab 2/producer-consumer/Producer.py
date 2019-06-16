import sys
import time
import random
from threading import Thread


class Producer(Thread):

    def __init__(self, buffer):
        Thread.__init__(self)
        self.buffer = buffer

    def run(self):
        random.seed(int(time.time()))

        for i in range(self.buffer.size()):
            item = random.randint(0, sys.maxint)
            self.buffer.put(item)
            print "Producer{} produce {}\n".format(self.name, item)
