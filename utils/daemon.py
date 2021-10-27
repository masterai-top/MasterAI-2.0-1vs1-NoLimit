#!/usr/bin/env python
# encoding: utf-8

import atexit
import logging
import os
import signal
import sys
import time


class DaemonBase:
    '''
    a generic daemon class.
    usage: subclass the CDaemon class and override the run() method
    stderr  表示错误日志文件绝对路径, 收集启动过程中的错误日志
    verbose 表示将启动运行过程中的异常错误信息打印到终端,便于调试,建议非调试模式下关闭, 默认为1, 表示开启
    save_path 表示守护进程pid文件的绝对路径
    '''

    def __init__(self, save_path,
                 stdin=os.devnull,
                 stdout=os.devnull,
                 stderr=os.devnull,
                 home_dir='.',
                 umask=0o22,
                 verbose=1):
        self.stdin = stdin
        self.stdout = stdout
        self.stderr = stderr
        self.pidfile = save_path
        self.home_dir = home_dir
        self.verbose = verbose
        self.umask = umask
        self.daemon_alive = True

    def daemonize(self):
        try:
            pid = os.fork()
            if pid > 0:
                sys.exit(0)
        except OSError as e:
            sys.stderr.write('fork #1 failed: %d (%s)\n' % (e.errno, e.strerror))
            sys.exit(1)

        os.chdir(self.home_dir)
        os.setsid()
        os.umask(self.umask)

        try:
            pid = os.fork()
            if pid > 0:
                sys.exit(0)
        except OSError as e:
            sys.stderr.write('fork #2 failed: %d (%s)\n' % (e.errno, e.strerror))
            sys.exit(1)

        sys.stdout.flush()
        sys.stderr.flush()

        si = open(self.stdin, 'r')
        so = open(self.stdout, 'a+')
        if self.stderr:
            se = open(self.stderr, 'a+', 0)
        else:
            se = so

        os.dup2(si.fileno(), sys.stdin.fileno())
        os.dup2(so.fileno(), sys.stdout.fileno())
        os.dup2(se.fileno(), sys.stderr.fileno())

        def sig_handler(signum, frame):
            self.daemon_alive = False

        signal.signal(signal.SIGTERM, sig_handler)
        signal.signal(signal.SIGINT, sig_handler)
        logging.info('daemon process started ...')

        atexit.register(self.del_pid)
        pid = str(os.getpid())
        open(self.pidfile, 'w+').write('%s\n' % pid)

    def get_pid(self):
        try:
            pf = open(self.pidfile, 'r')
            pid = int(pf.read().strip())
            pf.close()
        except IOError:
            pid = None
        except SystemExit:
            pid = None
        return pid

    def del_pid(self):
        if os.path.exists(self.pidfile):
            os.remove(self.pidfile)

    def start(self, *args, **kwargs):
        logging.info('ready to starting ......')
        # check for a pid file to see if the daemon already runs
        pid = self.get_pid()
        if pid:
            msg = 'pid file {} already exists, is it already running?'.format(self.pidfile)
            sys.stderr.write(msg)
            sys.exit(1)
        # start the daemon
        self.daemonize()
        self.run(*args, **kwargs)

    def stop(self):
        logging.info('stopping ...')
        pid = self.get_pid()
        if not pid:
            msg = 'pid file [{}] does not exist. Not running?'.format(self.pidfile)
            sys.stderr.write(msg)
            if os.path.exists(self.pidfile):
                os.remove(self.pidfile)
            return
        # try to kill the daemon process
        try:
            i = 0
            while 1:
                os.kill(pid, signal.SIGTERM)
                time.sleep(0.1)
                i = i + 1
                if i % 10 == 0:
                    os.kill(pid, signal.SIGHUP)
        except OSError as err:
            err = str(err)
            if err.find('No such process.') > 0:
                if os.path.exists(self.pidfile):
                    os.remove(self.pidfile)
            else:
                logging.error(str(err))
                sys.exit(1)
            logging.info('Stopped.')

    def restart(self, *args, **kwargs):
        self.stop()
        self.start(*args, **kwargs)

    def is_running(self):
        pid = self.get_pid()
        return pid and os.path.exists('/proc/%d' % pid)

    def run(self, *args, **kwargs):
        'NOTE: override the method in subclass'
        logging.error('base class run()')


class ProcessDaemon(DaemonBase):
    def __init__(self, name, save_path,
                 stdin=os.devnull,
                 stdout=os.devnull,
                 stderr=os.devnull,
                 home_dir='.',
                 umask=0o22,
                 verbose=1):
        super(ProcessDaemon, self).__init__(self, save_path, stdin, stdout, stderr, home_dir, umask, verbose)
        self.name = name

    def run(self, output_fn, **kwargs):
        fd = open(output_fn, 'w')
        while True:
            line = time.ctime() + '\n'
            fd.write(line)
            fd.flush()
            time.sleep(1)
        fd.close()


def test_daemon():
    help_msg = 'Usage: python {} <start|stop|restart|status>'.format(sys.argv[0])
    if len(sys.argv) != 2:
        logging.debug('{}'.format(help_msg))
        sys.exit(1)

    proc_name = 'rebel_train'  # 守护进程名称
    pid_fn = '/tmp/{}.pid'.format(proc_name)  # 守护进程pid文件的绝对路径
    log_fn = '/tmp/{}.log'.format(proc_name)  # 守护进程日志文件的绝对路径
    err_fn = '/tmp/{}.err'.format(proc_name)  # 守护进程启动过程中的错误日志
    pd = ProcessDaemon(proc_name, pid_fn, stderr=err_fn, verbose=1)
    if sys.argv[1] == 'start':
        pd.start(log_fn)
    elif sys.argv[1] == 'stop':
        pd.stop()
    elif sys.argv[1] == 'restart':
        pd.restart(log_fn)
    elif sys.argv[1] == 'status':
        if pd.is_running():
            logging.info('process [{}] is running.'.format(pd.get_pid()))
        else:
            logging.info('process [{}] stopped.'.format(pd.name))


if __name__ == '__main__':
    test_daemon()
