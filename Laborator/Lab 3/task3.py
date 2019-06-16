from threading import enumerate, current_thread, Event, Thread, Condition


class Master(Thread):
    # def __init__(self, max_work, work_available, result_available):
    def __init__(self, max_work, condition):
        Thread.__init__(self, name="Master")
        self.max_work = max_work

        # self.work_available = work_available
        # self.result_available = result_available

        self.work = False
        self.condition = condition

    def set_worker(self, worker):
        self.worker = worker

    def run(self):
        for i in xrange(self.max_work):
            with self.condition:
                # generate work
                self.work = i

                # notify worker
                # self.work_available.set()
                self.work = True
                self.condition.notify()

                # get result
                # self.result_available.wait()
                # self.result_available.clear()
                self.condition.wait()

                if self.get_work() + 1 != self.worker.get_result():
                    print "oops",
                print "%d -> %d" % (self.work, self.worker.get_result())

    def get_work(self):
        print " work {} thread {} \n".format(str(self), str(current_thread())),
        return self.work


class Worker(Thread):
    # def __init__(self, terminate, work_available, result_available):
    def __init__(self, terminate, condition):
        Thread.__init__(self, name="Worker")
        self.terminate = terminate

        # self.work_available = work_available
        # self.result_available = result_available

        self.condition = condition

    def set_master(self, master):
        self.master = master

    def run(self):
        while (True):
            with self.condition:
                # wait work
                # self.work_available.wait()
                # self.work_available.clear()
                if not self.master.work:
                    self.condition.wait()
                    self.master.work = False

                if (terminate.is_set()): break
                # generate result

                self.result = self.master.get_work() + 1
                # notify master
                # self.result_available.set()
                self.condition.notify()
                self.master.work = True

    def get_result(self):
        return self.result


if __name__ == "__main__":
    # create shared objects
    terminate = Event()

    cond = Condition()

    # work_available = Event()
    # result_available = Event()

    # start worker and master
    w = Worker(terminate, cond)
    m = Master(10, cond)
    w.set_master(m)
    m.set_worker(w)
    w.start()
    m.start()

    # wait for master
    m.join()

    # wait for worker
    with cond:
        terminate.set()
        cond.notifyAll()
    w.join()

    # print running threads for verification
    print enumerate()
