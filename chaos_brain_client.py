import os
import socket
import time
import datetime
import logging

from chaos_brain_message import *


class ChaosBrainClient:
    def __init__(self, addr, port):
        self.addr_ = addr
        self.port_ = port
        self.client_ = None
        self.head_ = NetHead()
        self.counter_ = 0
        self.registerMsg_ = AlgNodeRegisterReq()
        self.keepaliveMsg_ = AlgKeepAliveReq()
        self.decideMsg_ = AlgRobotActionReq()

    def connect(self):
        self.client_ = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.client_.connect((self.addr_, self.port_))
        return True

    def test1_(self):
        self.registerMsg_ = AlgNodeRegisterReq()
        self.registerMsg_.token = "123456"
        body = self.registerMsg_.SerializeToString()
        head = self.head_.hton(len(body), ALG_CMD_NODE_REGISER_REQ, 11111, 0)
        self.client_.send(head + body)

    def test2_(self):
        self.keepaliveMsg_ = AlgKeepAliveReq()
        self.keepaliveMsg_.timestamp = int(datetime.datetime.now().timestamp())
        body = self.keepaliveMsg_.SerializeToString()
        head = self.head_.hton(len(body), ALG_CMD_KEEPALIVE_REQ, 11111, 0)
        self.client_.send(head + body)

    def test3_(self):
        self.decideMsg_ = AlgRobotActionReq()
        self.decideMsg_.uid = 11111
        body = self.decideMsg_.SerializeToString()
        head = self.head_.hton(len(body), ALG_CMD_ROBOT_DECIDE_REQ, 11111, 0)
        self.client_.send(head + body)

    def testing(self):
        send_time = datetime.datetime.now().timestamp()
        idx = self.counter_ % 3
        if 0 == idx:
            self.test1_()
        elif 1 == idx:
            self.test2_()
        elif 2 == idx:
            self.test3_()
        self.counter_ = self.counter_ + 1
        logging.debug("send done..%f", float(send_time))
        #
        raw = self.client_.recv(8192)
        if not raw:
            return
        #
        recv_time = datetime.datetime.now().timestamp()
        logging.debug("response...%f", float(recv_time - send_time))


if __name__ == "__main__":
    formatter = "%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s"
    logging.basicConfig(level=logging.NOTSET, format=formatter)
    logging.info("Emulator startup...")

    try:
        c = ChaosBrainClient("10.10.10.162", 8888)
        c.connect()
        while True:
            c.testing()
            time.sleep(0.25)
    except:
        raise NotImplementedError
