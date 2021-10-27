import logging
import os
import socketserver
import threading
import time
import traceback

from chaos_brain_message import NetHead, ChaosBrainConnector
from proto.alg_pb2 import *


class ChaosBrianHandler(socketserver.StreamRequestHandler):
    def setup(self):
        super(ChaosBrianHandler, self).setup()
        self.hand_ = ChaosBrainConnector()
        self.head_ = NetHead()
        self.buffer_ = bytes()
        self.test_ = False
        logging.debug("client setup %s" % str(self.client_address))

    def handle(self):
        if False == self.test_:
            self.handle1()
        else:
            self.handle2()

    def finish(self):
        logging.debug("client is disconnect %s" % str(self.client_address))
        super().finish()

    def dispatch(self, cmd, uid, data):
        logging.debug("message: cmd=%d, uid=%d, data=%d, from=%s" % (
            cmd, uid, len(data), str(self.client_address)
        ))
        if cmd == ALG_CMD_NODE_REGISER_REQ:
            response = self.hand_.node_register_req(data)
            self.send(ALG_CMD_NODE_REGISER_RSP, uid, response, 0)
        elif cmd == ALG_CMD_KEEPALIVE_REQ:
            response = self.hand_.keep_alive_req(data)
            self.send(ALG_CMD_KEEPALIVE_RSP, uid, response, 0)
        elif cmd == ALG_CMD_ROBOT_DECIDE_REQ:
            response = self.hand_.robot_action_req(data)
            self.send(ALG_CMD_ROBOT_DECIDE_RSP, uid, response, 0)
        else:
            # self.hand_.AlgInvalidReq(data)
            # super(ChaosBrianHandler, self).finish()
            logging.error("error: unknown cmd (%d)" % cmd)

    def send(self, cmd, uid, data, error):
        try:
            head = self.head_.hton(
                n_len=len(data), n_cmd=cmd, n_uid=uid, n_err=error)
            cur_thread = threading.current_thread()
            response = head + data
            self.request.sendall(response)
            logging.debug(
                "send data succeed:  uid=%d, cmd=%d, len=%d to=%s," % (uid, cmd, len(data), str(self.client_address)))
        except:
            exc_type, exc_value, exc_obj = sys.exc_info()
            logging.error(
                "send data error: exception_type: \t%s,\nexception_value: \t%s,\nexception_object: \t%s,\n%s"
                % (exc_type, exc_value, exc_obj, str(self.client_address))
            )

    def handle1(self):
        while True:
            try:
                raw = self.request.recv(8192)
                if not raw:
                    break
                # message cache
                self.buffer_ = self.buffer_ + raw
                while True:
                    # message head
                    if len(self.buffer_) < NetHead.NUM_HEAD_BYTES:
                        break
                    head_data = self.buffer_[0: NetHead.NUM_HEAD_BYTES]
                    n_len, n_cmd, n_uid, n_err = self.head_.ntoh(head_data)
                    # logging.debug("head: len=%d, cmd=%d, uid=%d" % (n_len, n_cmd, n_uid))
                    # message body
                    if len(self.buffer_) < n_len:
                        break
                    body_end = NetHead.NUM_HEAD_BYTES + n_len
                    unpack_body = \
                        self.buffer_[NetHead.NUM_HEAD_BYTES: body_end]
                    self.buffer_ = self.buffer_[body_end:]
                    self.dispatch(n_cmd, n_uid, unpack_body)
            except:
                exc_type, exc_value, exc_obj = sys.exc_info()
                logging.fatal(
                    "exception_type: \t%s,\nexception_value: \t%s,\nexception_object: \t%s"
                    % (exc_type, exc_value, exc_obj)
                )
                traceback.print_exc()
                self.finish()
                break
            time.sleep(0.01)

    def handle2(self):
        while True:
            raw = self.request.recv(8192)
            if not raw:
                break
            try:
                data = str(raw, "ascii")
                cur_thread = threading.current_thread()
                response = bytes("{}: {}".format(
                    cur_thread.name, data), "ascii")
                logging.debug("received data from %s" % data)
                self.request.sendall(response)
            except:
                logging.error(
                    "Unexpected(2) error: %s, %s" %
                    (sys.exc_info()[0], str(raw))
                )
                self.finish()
                break


class ChaosBrainServer(socketserver.ThreadingTCPServer):
    allow_reuse_address = True
    daemon_threads = True

    def __init__(self, server_address, RequestHandlerClass):
        socketserver.ThreadingTCPServer.__init__(self, server_address, RequestHandlerClass)
