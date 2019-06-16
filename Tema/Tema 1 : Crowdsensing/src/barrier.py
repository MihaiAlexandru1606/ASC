"""
@author Computer Systems Architecture Course
Implementarea unei bariere reutilizaabile luata din lab-ul 3.
"""
from threading import Condition


class ReusableBarrierCond(object):
    """ Bariera reentranta, implementata folosind o variabila conditie """

    def __init__(self, num_threads):
        """
        Construnctor

        :type : integer
        :param num_threads: numarul de thread-uri
        """
        self.num_threads = num_threads
        self.count_threads = self.num_threads
        # blocheaza/deblocheaza thread-urile
        self.cond = Condition()
        # protejeaza modificarea contorului

    def wait(self):
        """
        Asteapta ca toate thread-urile sa ajunga la bariera si sa treaca

        """
        # intra in regiunea critica
        self.cond.acquire()
        self.count_threads -= 1
        if self.count_threads == 0:
            # deblocheaza toate thread-urile
            self.cond.notify_all()
            self.count_threads = self.num_threads
        else:
            # blocheaza thread-ul eliberand in acelasi timp lock-ul
            self.cond.wait()
        # iese din regiunea critica
        self.cond.release()
