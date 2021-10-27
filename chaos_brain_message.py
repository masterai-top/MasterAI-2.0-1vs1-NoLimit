from proto.alg_pb2 import *
# from robot.gto_mc.bot import get_game_state, do_action
from robot.rebel.bot import get_game_state, do_action


class NetHead:
    """
    ALG_CMD
    ALG_CMD_NOT_DEFINE              : 0,
    ALG_CMD_ROBOT_DECIDE_REQ        : 10001,
    ALG_CMD_ROBOT_DECIDE_RSP        : 10002,
    ALG_CMD_ROBOT_KEEPALIVE_REQ     : 10003,
    ALG_CMD_ROBOT_KEEPALIVE_RSP     : 10004,
    ALG_CMD_MAX_VALUE               : 11111,

    ALG_ERR
    ALG_SUCC                        : 0,        # 成功
    ALG_FAIL                        : 1,        # 失败
    """
    NUM_HEAD_BYTES = 20

    def hton(self, n_len, n_cmd, n_uid, n_err):
        n_len = self.htonl(n_len + self.NUM_HEAD_BYTES)
        n_cmd = self.htonl(n_cmd)
        n_uid = self.htonl64(n_uid)
        n_err = self.htonl(n_err)
        buf = n_len + n_cmd + n_uid + n_err
        return buf

    def ntoh(self, buf):
        n_len = self.ntohl(buf[: 4]) - self.NUM_HEAD_BYTES
        n_cmd = self.ntohl(buf[4: 8])
        n_uid = self.ntohl(buf[8: 16])
        n_err = self.ntohl(buf[16: 20])
        return n_len, n_cmd, n_uid, n_err

    @staticmethod
    def htonl(x):
        return x.to_bytes(4, "big")

    @staticmethod
    def htonl64(x):
        return x.to_bytes(8, "big")

    @staticmethod
    def ntohl(x):
        return int.from_bytes(x, "big")


class ChaosBrainConnector:
    """communicate with aim server
    using pb proto
    """

    def __init__(self):
        self._net_head = NetHead()
        self._register = AlgNodeRegisterReq()
        self._keepalive = AlgKeepAliveReq()
        self._robot_action = AlgRobotActionReq()

    def node_register_req(self, data):
        """节点注册"""
        self._register = AlgNodeRegisterReq()
        self._register.ParseFromString(data)
        pb_obj = AlgNodeRegisterRsp()
        pb_obj.result = 0
        buf = pb_obj.SerializeToString()
        return buf

    def keep_alive_req(self, data):
        """连接心跳"""
        self._keepalive.ParseFromString(data)
        pb_obj = AlgKeepAliveRsp()
        pb_obj.timestamp = self._keepalive.timestamp
        buf = pb_obj.SerializeToString()
        return buf

    def robot_action_req(self, data):
        """机器决策"""
        self._robot_action = AlgRobotActionReq()
        self._robot_action.ParseFromString(data)
        # to do
        actions, state = get_game_state(self._robot_action)
        action, action_size = do_action(actions, state)
        pb_obj = AlgRobotActionRsp()
        pb_obj.result = 0
        pb_obj.uid = self._robot_action.uid
        pb_obj.action = action
        pb_obj.bet_size = action_size
        pb_obj.runGameID = self._robot_action.runGameID
        pb_obj.action_num = self._robot_action.action_num
        buf = pb_obj.SerializeToString()
        return buf
