import zmq
import logging
from ipc.config import *

if __name__ == '__main__':
    logging.basicConfig(level=logging.NOTSET,
                        format='%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s')
    context = zmq.Context()
    socket = context.socket(zmq.DEALER)
    socket.setsockopt(zmq.RCVTIMEO, 1000)
    socket.connect(worker_connect_addr)
    socket.send_multipart([b"heart", b""])
    while True:
        try:
            client_addr, message = socket.recv_multipart()
        except Exception as e:
            logging.error("{}".format(e))
            socket.send_multipart([b"heart", b""])
            continue
        logging.debug("received from broker{}: {}".format(client_addr, message))
        socket.send_multipart([client_addr, b"world"])
