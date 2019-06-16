import sys
import random
import time
from threading import Thread, Lock


class Filozof(Thread):

    def __init__(self, left_fork, right_fork, type):
        Thread.__init__(self)
        self.left_fork = left_fork
        self.right_fork = right_fork
        self.type = type

    def run(self):
        print "Think"
        random.seed(int(time.time()))
        time.sleep(random.randint(0, sys.maxint) % 10)
        print "End think"

        if self.type == 0:
            self.left_fork.acquire()
            self.right_fork.acquire()

            print "Eat"
            random.seed(int(time.time()))
            time.sleep(random.randint(0, sys.maxint) % 10)
            print "End eat {}", self.name

            self.left_fork.release()
            self.right_fork.release()

        elif self.type == 1:
            self.right_fork.acquire()
            self.left_fork.acquire()

            print "Eat"
            random.seed(int(time.time()))
            time.sleep(random.randint(0, sys.maxint) % 10)
            print "End eat"

            self.right_fork.release()
            self.left_fork.release()


if __name__ == "__main__":
    number = int(sys.stdin.readline().strip())

    fork = [Lock() for i in xrange(number)]
    philosoph = [Filozof(fork[i], fork[i + 1], 1) for i in xrange(0, number - 1)]
    philosoph.append(Filozof(fork[number - 1], fork[0], 0))

    for i in xrange(0, number):
        philosoph[i].start()

    for i in xrange(0, number):
        philosoph[i].join()