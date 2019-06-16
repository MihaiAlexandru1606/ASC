"""
Implementati o propagare ciclica de tip gossiping folosind bariere. 
  * Se considera N noduri (obiecte de tip Thread), cu indecsi 0...N-1.
  * Fiecare nod tine o valoare generata random.
  * Calculati valoarea minima folosind urmatorul procedeu:
     * nodurile ruleaza in cicluri

     * intr-un ciclu, fiecare nod comunica cu un subset de alte noduri pentru a
       obtine valoarea acestora si a o compara cu a sa
       * ca subset considerati random 3 noduri din lista de noduri primita in
        constructor si obtineti valoarea acestora (metoda get_value)

     * dupa terminarea unui ciclu, fiecare nod va avea ca valoare minimul
       obtinut in ciclul anterior

     * la finalul iteratiei toate nodurile vor contine valoarea minima

  * Folositi una din barierele reentrante din modulul barrier.
  * Pentru a simplifica implementarea, e folosit un numar fix de cicluri,
    negarantand astfel convergenta tutoror nodurilor la acelasi minim.
"""
import barrier
import sys
import random

from threading import *

random.seed(0)  # genereaza tot timpul aceeasi secventa pseudo-random


class Node(Thread):
    # TODO Node trebuie sa fie Thread

    def __init__(self, node_id, all_nodes, num_iter, barrier):
        # TODO nodurile trebuie sa foloseasca un obiect bariera
        Thread.__init__(self)
        self.node_id = node_id
        self.all_nodes = all_nodes
        self.num_iter = num_iter
        self.value = random.randint(1, 1000)
        self.barrier = barrier

    def get_value(self):
        return self.value

    def run(self):
        for i in range(0, self.num_iter):
            neighbour = []
            compare_val = self.value

            for j in range(0, 3):
                #neighbour = random.choice(3)
                index = random.randint(0, len(self.all_nodes) - 1)
                while True:
                    if index not in neighbour:
                        break
                    index = random.randint(0, len(self.all_nodes) - 1)
                neighbour.append(index)
            for index_neighbour in neighbour:
                compare_val = min(compare_val, self.all_nodes[index_neighbour].get_value())

            self.barrier.wait()
            self.value = compare_val
            self.barrier.wait()
        print "Final value is {}\n".format(self.value),


if __name__ == "__main__":
    if len(sys.argv) == 2:
        num_threads = int(sys.argv[1])
    else:
        print "Usage: python " + sys.argv[0] + " num_nodes"
        sys.exit(0)

    num_iter = 10  # numar iteratii/cicluri algoritm

    barrier = barrier.ReusableBarrierCond(num_threads)

    threads = []
    for i in range(num_threads):
        threads.append(Node(i, threads, num_iter, barrier))

    for node in threads:
        node.start()

    for t in threads:
        t.join()
