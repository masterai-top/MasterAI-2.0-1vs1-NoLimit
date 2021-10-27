import logging
import logging.handlers
import os
import signal
import sys
import threading
import torch

from chaos_brain_server import ChaosBrainServer, ChaosBrianHandler

LOG_NAME = "chaos_brain_server"


def info_handler(signum, frame):
    logging.info("signal received: %d, loglevel is info" % signum)
    logging.getLogger().setLevel(logging.INFO)


def debug_handler(signum, frame):
    logging.info("signal received: %d, loglevel is debug" % signum)
    logging.getLogger().setLevel(logging.DEBUG)


def error_handler(signum, frame):
    logging.info("signal received: %d, loglevel is error" % signum)
    logging.getLogger().setLevel(logging.ERROR)


def fatal_handler(signum, frame):
    logging.info("signal received: %d, loglevel is critical" % signum)
    logging.getLogger().setLevel(logging.FATAL)


def main(argv):
    print(torch.__version__)
    print(torch.version.cuda)
    print(torch.cuda.is_available())
    print(torch.cuda.device_count())

    try:
        port = int(argv[1])
    except:
        port = 8888
    if sys.platform != "win32":
        signal.signal(signal.SIGUSR1, info_handler)
        signal.signal(signal.SIGUSR2, debug_handler)
        signal.signal(signal.SIGPIPE, signal.SIG_IGN)
    log_dir = "./log"
    if not os.path.isdir(log_dir):
        os.mkdir(log_dir)
    logger = logging.getLogger(LOG_NAME)
    formatter = "%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s"
    file_handler = logging.handlers.TimedRotatingFileHandler(
        filename=os.path.join(log_dir, "chaos_brain_srv.log"),
        encoding="utf-8",
        when="D"
    )
    stream_handler = logging.StreamHandler()
    logging.basicConfig(
        level=logging.INFO,
        format=formatter,
        handlers=[file_handler, stream_handler]
    )

    host, port = "0.0.0.0", port
    logging.info("Server({}:{}) running(fork={}) ...".format(host, port, hasattr(os, 'fork')))

    with ChaosBrainServer((host, port), ChaosBrianHandler) as server:
        server.request_queue_size = 10000
        server.serve_forever(0.01)
        server.server_close()
        logging.info("Server exited %s:%d" % (host, port))


if __name__ == "__main__":
    main(sys.argv)
