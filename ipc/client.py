import zmq
import logging
from ipc.config import *

if __name__ == '__main__':
    logging.basicConfig(level=logging.NOTSET,
                        format='%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s')
    ctx = zmq.Context.instance()
    socket = ctx.socket(zmq.DEALER)
    socket.connect(client_connect_addr)
    while True:
        socket.send(b"hello")
        msg = socket.recv()
        logging.debug("rsp={}".format(msg))
