"""
Modulul implementeaza un thread pool, folosind modelul producator-consumator
Niculescu Mihai Alexandru
"""
from Queue import Queue
from threading import Thread


class TreadPool(object):
    """
    Clasa reazalizaza planificarea celor 8 thread-uri.
    """

    def __init__(self, protect_data):
        """
        Constructor.

        :type : dictionary : (Integer, Lock)
        :param protect_data: lock-urile pentru fiecare locatie,
                            doar un thread poate sa modifice date la o anunumita locatie
        """
        self.__max_worker = 8
        self.__tasks = Queue()

        self.devices = None
        self.protect_data = protect_data

        # pornirea thread-urilor
        self.__workers = [WorkerThread(self) for _ in range(self.__max_worker)]
        for i in xrange(self.__max_worker):
            self.__workers[i].start()

    def start_work(self, devices, scripts):
        """
        Setarea device-urilor la un timepoint, si adaugarea script-urilor primte
            pana la acel timepoint

        :type : List of Device
        :param devices lista de device-uri pentru timepoint-ul curent

        :type : List of Script
        :param : scripts : script-urile existente pana la timepoint-ul curent
        """
        self.devices = devices

        for script in scripts:
            self.__tasks.put(script)

    def end_work(self):
        """
        Teminarea executie celor 8 thread-uri.
        """

        # trimiterea mesajului care indica finalizarea procesari.
        for _ in xrange(self.__max_worker):
            self.__tasks.put((None, None))

        for i in xrange(self.__max_worker):
            self.__workers[i].join()

    def add_task(self, script, location):
        """
        Adauga un nou task de executat.

        @type script: Script
        @param script: the script to execute from now on at each timepoint; None if the
            current timepoint has ended

        @type location: Integer
        @param location: the location for which the script is interested in
        """
        self.__tasks.put((script, location))

    def get_task(self):
        """
        Retuneza task, scriptul, care il va executa un worker.

        :return: (script, location)
        """
        return self.__tasks.get()

    def worker_task_done(self):
        """
        Semaliarea faptului ca un task a fost finalizat de catre un worker.
        """
        self.__tasks.task_done()

    def join(self):
        """
        Teriminarea tuturol script-urilor de executa pentru un timepoint.

        """
        self.__tasks.join()


class WorkerThread(Thread):
    """
    Clasa care executa, relucreaza un script.
    """
    def __init__(self, master):
        """
        Constructor

        :type : TreadPool
        :param master: thread pool-ul de care tine acest thread.
        """
        Thread.__init__(self)
        self.master = master

    def run(self):

        while True:
            (script, location) = self.master.get_task()

            # mesajul care indica sfarsitul executiei complete
            if script is None and location is None:
                break

            # executarea script-ului
            with self.master.protect_data[location]:
                data_location = []

                # colectarea de date
                for device in self.master.devices:
                    data = device.get_data(location)
                    if data is not None:
                        data_location.append(data)

                if data_location != []:
                    # rezultatul rulari scriptului
                    rezult = script.run(data_location)

                    # updatarea tutorol device-urilor
                    for device in self.master.devices:
                        device.set_data(location, rezult)

            self.master.worker_task_done()
