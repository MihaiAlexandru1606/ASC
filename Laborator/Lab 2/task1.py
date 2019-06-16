"""
    Basic thread handling exercise:

    Use the Thread class to create and run more than 10 threads which print their name and a random
    number they receive as argument. The number of threads must be received from the command line.

    e.g. Hello, I'm Thread-96 and I received the number 42

"""
import sys
import random
import time
from threading import Thread


class MyThread(Thread):

    def __init__(self, number):
        Thread.__init__(self)
        self.number = number

    def run(self):
        print "Hello World! My name is ", self.name, " and my magic number is : ", self.number


if __name__ == "__main__":

    if sys.argv.__len__() < 2:
        print "Usage : thread.py [numberThread]"
        sys.exit(1)

    threads = []
    random.seed(int(time.time()))

    for i in xrange(0, int(sys.argv[1])):
        threads.append(MyThread(random.randint(0, sys.maxint)))

    for i in range(int(sys.argv[1])):
        threads[i].start()

    for i in xrange(0, int(sys.argv[1])):
        threads[i].join()
