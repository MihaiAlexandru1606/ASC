"""
Niculescu Mihai
Modulul ce reprezinta un device.
"""

from threading import Event, Thread, Lock
from barrier import ReusableBarrierCond
from threadpool import TreadPool


class Device(object):
    """
    Class that represents a device.
    """

    def __init__(self, device_id, sensor_data, supervisor):
        """
        Constructor.

        @type device_id: Integer
        @param device_id: the unique id of this node; between 0 and N-1

        @type sensor_data: List of (Integer, Float)
        @param sensor_data: a list containing (location, data) as measured by this device

        @type supervisor: Supervisor
        @param supervisor: the testing infrastructure's control and validation component
        """
        self.device_id = device_id
        self.sensor_data = sensor_data
        self.supervisor = supervisor

        self.timepoint_done = Event()
        # indica daca un device a inceput unui timepoint
        # permite primirea de script-uri dupa ce primiste vecini si isi seteaza cele 8 thread-uri
        self.start = Event()

        self.scripts = []
        self.barrier = None
        # pentru fiecare locatie o sa existe un lock, asta ca sa asigure ca doar un device
        # poate sa prelucreze datele
        self.protect_data = None
        self.thread = None

    def __str__(self):
        """
        Pretty prints this device.

        @rtype: String
        @return: a string containing the id of this device
        """
        return "Device %d" % self.device_id

    def setup_devices(self, devices):
        """
        Setup the devices before simulation begins.

        @type devices: List of Device
        @param devices: list containing all devices
        """
        # device-ul cu id : 0 va seta pentru toate bariera si lock-urile
        if self.device_id == 0:
            barrier = ReusableBarrierCond(len(devices))
            protect_data = dict()

            # identificarea tuturilor locatilor pentru care exista date
            for device in devices:
                for (location, _) in device.sensor_data.iteritems():
                    if location not in protect_data:
                        protect_data[location] = Lock()

            for device in devices:
                device.barrier = barrier
                device.protect_data = protect_data
                device.thread = DeviceThread(device)
                device.thread.start()

    def assign_script(self, script, location):
        """
        Provide a script for the device to execute.

        @type script: Script
        @param script: the script to execute from now on at each timepoint; None if the
            current timepoint has ended

        @type location: Integer
        @param location: the location for which the script is interested in
        """
        # asteapta pana a primit vecini si thread pool este poate sa primiesca noi script-uri
        self.start.wait()
        if script is not None:
            self.thread.add_task(script, location)
            self.scripts.append((script, location))
        else:
            self.timepoint_done.set()

    def get_data(self, location):
        """
        Returns the pollution value this device has for the given location.

        @type location: Integer
        @param location: a location for which obtain the data

        @rtype: Float
        @return: the pollution value
        """
        return self.sensor_data[location] if location in self.sensor_data else None

    def set_data(self, location, data):
        """
        Sets the pollution value stored by this device for the given location.

        @type location: Integer
        @param location: a location for which to set the data

        @type data: Float
        @param data: the pollution value
        """

        if location in self.sensor_data:
            self.sensor_data[location] = data

    def shutdown(self):
        """
        Instructs the device to shutdown (terminate all threads). This method
        is invoked by the tester. This method must block until all the threads
        started by this device terminate.
        """
        self.thread.join()


class DeviceThread(Thread):
    """
    Class that implements the device's worker thread.
    """

    def __init__(self, device):
        """
        Constructor.

        @type device: Device
        @param device: the device which owns this thread
        """
        Thread.__init__(self, name="Device Thread %d" % device.device_id)
        self.device = device
        # utilizat ca un device sa ruleze in paralel script-urile
        self.thread_pool = TreadPool(device.protect_data)

    def add_task(self, script, location):
        """
        Adauga un nou task de executat.

        @type script: Script
        @param script: the script to execute from now on at each timepoint; None if the
            current timepoint has ended

        @type location: Integer
        @param location: the location for which the script is interested in
        """
        self.thread_pool.add_task(script, location)

    def run(self):

        while True:
            # get the current neighbourhood
            neighbours = self.device.supervisor.get_neighbours()

            if neighbours is None:
                self.thread_pool.end_work()
                break

            neighbours.append(self.device)
            self.thread_pool.start_work(neighbours, self.device.scripts)
            # indica faptul ca un device poate sa primiesca noi script-uri
            self.device.start.set()

            self.device.timepoint_done.wait()
            # este sa se termine de executat toate task-urilre pentru respectivul timepoint
            self.thread_pool.join()
            self.device.timepoint_done.clear()
            self.device.start.clear()
            # astept ca toate device-urile sa termine timepoint-ul curent
            self.device.barrier.wait()
