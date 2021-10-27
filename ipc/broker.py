import zmq
import time
import logging
from collections import OrderedDict

from ipc.config import *

context = zmq.Context.instance()
frontend = context.socket(zmq.ROUTER)
frontend.bind(frontend_listen_addr)
frontend.setsockopt(zmq.RCVHWM, 100)
backend = context.socket(zmq.ROUTER)
backend.bind(backend_listen_addr)
backend.setsockopt(zmq.RCVHWM, 100)

workers = OrderedDict()
clients = {}
msg_cache = []
poll = zmq.Poller()
poll.register(backend, zmq.POLLIN)
poll.register(frontend, zmq.POLLIN)

if __name__ == '__main__':
    logging.basicConfig(level=logging.NOTSET,
                        format='%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s')

    while True:
        socks = dict(poll.poll(512))
        now = time.time()
        #
        if backend in socks and socks[backend] == zmq.POLLIN:
            worker_addr, client_addr, response = backend.recv_multipart()
            workers[worker_addr] = time.time()
            if client_addr in clients:
                frontend.send_multipart([client_addr, response])
                clients.pop(client_addr)
            else:
                logging.debug('{} {} {}'.format(worker_addr, client_addr, response))
        while len(msg_cache) > 0 and len(workers) > 0:
            worker_addr, t = workers.popitem()
            if t - now > 1:
                continue
            msg = msg_cache.pop(0)
            backend.send_multipart([worker_addr, msg[0], msg[1]])
        #
        if frontend in socks and socks[frontend] == zmq.POLLIN:
            client_addr, request = frontend.recv_multipart()
            clients[client_addr] = 1
            while len(workers) > 0:
                worker_addr, t = workers.popitem()
                if t - now > 1:
                    continue
                backend.send_multipart([worker_addr, client_addr, request])
                break
            else:
                msg_cache.append([client_addr, request])
