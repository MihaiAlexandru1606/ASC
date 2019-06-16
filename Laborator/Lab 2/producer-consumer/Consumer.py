from threading import Thread


class Consumer(Thread):

    def __init__(self, buffer):
        Thread.__init__(self)
        self.buffer = buffer

    def run(self):
        for i in range(self.buffer.size()):
            item = self.buffer.get()
            print "Consumer consume {}\n".format(item)
