import sys
import Producer
import Consumer
import Buffer

if __name__ == "__main__":

    if sys.argv.__len__() < 4:
        print "Usage : python main.py [no_consumer] [no_producer] [buffer_size]"
        sys.exit(1)

    buffer = Buffer.Buffer(int(sys.argv[3]))
    consumers = [Consumer.Consumer(buffer) for i in range(int(sys.argv[1]))]
    producers = [Producer.Producer(buffer) for i in range(int(sys.argv[2]))]

    for i in consumers:
        i.start()
    for i in producers:
        i.start()

    for i in consumers:
        i.join()
    for i in producers:
        i.join()
