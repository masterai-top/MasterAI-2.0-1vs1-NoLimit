# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: alg.proto

import sys
_b=sys.version_info[0]<3 and (lambda x:x) or (lambda x:x.encode('latin1'))
from google.protobuf.internal import enum_type_wrapper
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from google.protobuf import reflection as _reflection
from google.protobuf import symbol_database as _symbol_database
from google.protobuf import descriptor_pb2
# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()




DESCRIPTOR = _descriptor.FileDescriptor(
  name='alg.proto',
  package='alg',
  syntax='proto3',
  serialized_pb=_b('\n\talg.proto\x12\x03\x61lg\"#\n\x12\x41lgNodeRegisterReq\x12\r\n\x05token\x18\x01 \x01(\t\"$\n\x12\x41lgNodeRegisterRsp\x12\x0e\n\x06result\x18\x01 \x01(\x05\"$\n\x0f\x41lgKeepAliveReq\x12\x11\n\ttimestamp\x18\x01 \x01(\x05\"$\n\x0f\x41lgKeepAliveRsp\x12\x11\n\ttimestamp\x18\x01 \x01(\x05\"\x9a\x04\n\x11\x41lgRobotActionReq\x12\x0b\n\x03uid\x18\x01 \x01(\x03\x12\x13\n\x0bsmall_blind\x18\x02 \x01(\x05\x12\x11\n\tbig_blind\x18\x03 \x01(\x05\x12\x11\n\tis_dealer\x18\x04 \x01(\x08\x12\x11\n\thole_card\x18\x05 \x03(\x05\x12\x12\n\nboard_card\x18\x06 \x03(\x05\x12\x0e\n\x06street\x18\x07 \x01(\x05\x12\x13\n\x0bself_bet_to\x18\x08 \x01(\x05\x12\x13\n\x0boppo_bet_to\x18\t \x01(\x05\x12\x17\n\x0fself_init_chips\x18\n \x01(\x05\x12\x17\n\x0foppo_init_chips\x18\x0b \x01(\x05\x12\x1a\n\x12self_street_bet_to\x18\x0c \x01(\x05\x12\x1a\n\x12oppo_street_bet_to\x18\r \x01(\x05\x12\x1a\n\x12self_allow_actions\x18\x0e \x01(\x05\x12\x11\n\trunGameID\x18\x0f \x01(\x03\x12;\n\x0b\x61\x63tion_list\x18\x11 \x03(\x0b\x32&.alg.AlgRobotActionReq.ActionListEntry\x12\x12\n\naction_num\x18\x12 \x01(\x05\x1a\x1c\n\nActionList\x12\x0e\n\x06\x61\x63tion\x18\x01 \x03(\x05\x1aT\n\x0f\x41\x63tionListEntry\x12\x0b\n\x03key\x18\x01 \x01(\x05\x12\x30\n\x05value\x18\x02 \x01(\x0b\x32!.alg.AlgRobotActionReq.ActionList:\x02\x38\x01\"y\n\x11\x41lgRobotActionRsp\x12\x0e\n\x06result\x18\x01 \x01(\x03\x12\x0b\n\x03uid\x18\x02 \x01(\x03\x12\x0e\n\x06\x61\x63tion\x18\x03 \x01(\x05\x12\x10\n\x08\x62\x65t_size\x18\x04 \x01(\x05\x12\x11\n\trunGameID\x18\x05 \x01(\x03\x12\x12\n\naction_num\x18\x06 \x01(\x05*\xdb\x01\n\x07\x41LG_CMD\x12\x16\n\x12\x41LG_CMD_NOT_DEFINE\x10\x00\x12\x1e\n\x18\x41LG_CMD_NODE_REGISER_REQ\x10\xe1\xd4\x03\x12\x1e\n\x18\x41LG_CMD_NODE_REGISER_RSP\x10\xe2\xd4\x03\x12\x1b\n\x15\x41LG_CMD_KEEPALIVE_REQ\x10\xe3\xd4\x03\x12\x1b\n\x15\x41LG_CMD_KEEPALIVE_RSP\x10\xe4\xd4\x03\x12\x1e\n\x18\x41LG_CMD_ROBOT_DECIDE_REQ\x10\xe5\xd4\x03\x12\x1e\n\x18\x41LG_CMD_ROBOT_DECIDE_RSP\x10\xe6\xd4\x03*%\n\x07\x41LG_ERR\x12\x0c\n\x08\x41LG_SUCC\x10\x00\x12\x0c\n\x08\x41LG_FAIL\x10\x01\x42\x03\x80\x01\x00\x62\x06proto3')
)

_ALG_CMD = _descriptor.EnumDescriptor(
  name='ALG_CMD',
  full_name='alg.ALG_CMD',
  filename=None,
  file=DESCRIPTOR,
  values=[
    _descriptor.EnumValueDescriptor(
      name='ALG_CMD_NOT_DEFINE', index=0, number=0,
      options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='ALG_CMD_NODE_REGISER_REQ', index=1, number=60001,
      options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='ALG_CMD_NODE_REGISER_RSP', index=2, number=60002,
      options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='ALG_CMD_KEEPALIVE_REQ', index=3, number=60003,
      options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='ALG_CMD_KEEPALIVE_RSP', index=4, number=60004,
      options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='ALG_CMD_ROBOT_DECIDE_REQ', index=5, number=60005,
      options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='ALG_CMD_ROBOT_DECIDE_RSP', index=6, number=60006,
      options=None,
      type=None),
  ],
  containing_type=None,
  options=None,
  serialized_start=834,
  serialized_end=1053,
)
_sym_db.RegisterEnumDescriptor(_ALG_CMD)

ALG_CMD = enum_type_wrapper.EnumTypeWrapper(_ALG_CMD)
_ALG_ERR = _descriptor.EnumDescriptor(
  name='ALG_ERR',
  full_name='alg.ALG_ERR',
  filename=None,
  file=DESCRIPTOR,
  values=[
    _descriptor.EnumValueDescriptor(
      name='ALG_SUCC', index=0, number=0,
      options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='ALG_FAIL', index=1, number=1,
      options=None,
      type=None),
  ],
  containing_type=None,
  options=None,
  serialized_start=1055,
  serialized_end=1092,
)
_sym_db.RegisterEnumDescriptor(_ALG_ERR)

ALG_ERR = enum_type_wrapper.EnumTypeWrapper(_ALG_ERR)
ALG_CMD_NOT_DEFINE = 0
ALG_CMD_NODE_REGISER_REQ = 60001
ALG_CMD_NODE_REGISER_RSP = 60002
ALG_CMD_KEEPALIVE_REQ = 60003
ALG_CMD_KEEPALIVE_RSP = 60004
ALG_CMD_ROBOT_DECIDE_REQ = 60005
ALG_CMD_ROBOT_DECIDE_RSP = 60006
ALG_SUCC = 0
ALG_FAIL = 1



_ALGNODEREGISTERREQ = _descriptor.Descriptor(
  name='AlgNodeRegisterReq',
  full_name='alg.AlgNodeRegisterReq',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='token', full_name='alg.AlgNodeRegisterReq.token', index=0,
      number=1, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=_b("").decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=18,
  serialized_end=53,
)


_ALGNODEREGISTERRSP = _descriptor.Descriptor(
  name='AlgNodeRegisterRsp',
  full_name='alg.AlgNodeRegisterRsp',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='result', full_name='alg.AlgNodeRegisterRsp.result', index=0,
      number=1, type=5, cpp_type=1, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=55,
  serialized_end=91,
)


_ALGKEEPALIVEREQ = _descriptor.Descriptor(
  name='AlgKeepAliveReq',
  full_name='alg.AlgKeepAliveReq',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='timestamp', full_name='alg.AlgKeepAliveReq.timestamp', index=0,
      number=1, type=5, cpp_type=1, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=93,
  serialized_end=129,
)


_ALGKEEPALIVERSP = _descriptor.Descriptor(
  name='AlgKeepAliveRsp',
  full_name='alg.AlgKeepAliveRsp',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='timestamp', full_name='alg.AlgKeepAliveRsp.timestamp', index=0,
      number=1, type=5, cpp_type=1, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=131,
  serialized_end=167,
)


_ALGROBOTACTIONREQ_ACTIONLIST = _descriptor.Descriptor(
  name='ActionList',
  full_name='alg.AlgRobotActionReq.ActionList',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='action', full_name='alg.AlgRobotActionReq.ActionList.action', index=0,
      number=1, type=5, cpp_type=1, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=594,
  serialized_end=622,
)

_ALGROBOTACTIONREQ_ACTIONLISTENTRY = _descriptor.Descriptor(
  name='ActionListEntry',
  full_name='alg.AlgRobotActionReq.ActionListEntry',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='key', full_name='alg.AlgRobotActionReq.ActionListEntry.key', index=0,
      number=1, type=5, cpp_type=1, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='value', full_name='alg.AlgRobotActionReq.ActionListEntry.value', index=1,
      number=2, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  options=_descriptor._ParseOptions(descriptor_pb2.MessageOptions(), _b('8\001')),
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=624,
  serialized_end=708,
)

_ALGROBOTACTIONREQ = _descriptor.Descriptor(
  name='AlgRobotActionReq',
  full_name='alg.AlgRobotActionReq',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='uid', full_name='alg.AlgRobotActionReq.uid', index=0,
      number=1, type=3, cpp_type=2, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='small_blind', full_name='alg.AlgRobotActionReq.small_blind', index=1,
      number=2, type=5, cpp_type=1, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='big_blind', full_name='alg.AlgRobotActionReq.big_blind', index=2,
      number=3, type=5, cpp_type=1, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='is_dealer', full_name='alg.AlgRobotActionReq.is_dealer', index=3,
      number=4, type=8, cpp_type=7, label=1,
      has_default_value=False, default_value=False,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='hole_card', full_name='alg.AlgRobotActionReq.hole_card', index=4,
      number=5, type=5, cpp_type=1, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='board_card', full_name='alg.AlgRobotActionReq.board_card', index=5,
      number=6, type=5, cpp_type=1, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='street', full_name='alg.AlgRobotActionReq.street', index=6,
      number=7, type=5, cpp_type=1, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='self_bet_to', full_name='alg.AlgRobotActionReq.self_bet_to', index=7,
      number=8, type=5, cpp_type=1, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='oppo_bet_to', full_name='alg.AlgRobotActionReq.oppo_bet_to', index=8,
      number=9, type=5, cpp_type=1, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='self_init_chips', full_name='alg.AlgRobotActionReq.self_init_chips', index=9,
      number=10, type=5, cpp_type=1, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='oppo_init_chips', full_name='alg.AlgRobotActionReq.oppo_init_chips', index=10,
      number=11, type=5, cpp_type=1, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='self_street_bet_to', full_name='alg.AlgRobotActionReq.self_street_bet_to', index=11,
      number=12, type=5, cpp_type=1, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='oppo_street_bet_to', full_name='alg.AlgRobotActionReq.oppo_street_bet_to', index=12,
      number=13, type=5, cpp_type=1, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='self_allow_actions', full_name='alg.AlgRobotActionReq.self_allow_actions', index=13,
      number=14, type=5, cpp_type=1, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='runGameID', full_name='alg.AlgRobotActionReq.runGameID', index=14,
      number=15, type=3, cpp_type=2, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='action_list', full_name='alg.AlgRobotActionReq.action_list', index=15,
      number=17, type=11, cpp_type=10, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='action_num', full_name='alg.AlgRobotActionReq.action_num', index=16,
      number=18, type=5, cpp_type=1, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
  ],
  extensions=[
  ],
  nested_types=[_ALGROBOTACTIONREQ_ACTIONLIST, _ALGROBOTACTIONREQ_ACTIONLISTENTRY, ],
  enum_types=[
  ],
  options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=170,
  serialized_end=708,
)


_ALGROBOTACTIONRSP = _descriptor.Descriptor(
  name='AlgRobotActionRsp',
  full_name='alg.AlgRobotActionRsp',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='result', full_name='alg.AlgRobotActionRsp.result', index=0,
      number=1, type=3, cpp_type=2, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='uid', full_name='alg.AlgRobotActionRsp.uid', index=1,
      number=2, type=3, cpp_type=2, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='action', full_name='alg.AlgRobotActionRsp.action', index=2,
      number=3, type=5, cpp_type=1, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='bet_size', full_name='alg.AlgRobotActionRsp.bet_size', index=3,
      number=4, type=5, cpp_type=1, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='runGameID', full_name='alg.AlgRobotActionRsp.runGameID', index=4,
      number=5, type=3, cpp_type=2, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='action_num', full_name='alg.AlgRobotActionRsp.action_num', index=5,
      number=6, type=5, cpp_type=1, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=710,
  serialized_end=831,
)

_ALGROBOTACTIONREQ_ACTIONLIST.containing_type = _ALGROBOTACTIONREQ
_ALGROBOTACTIONREQ_ACTIONLISTENTRY.fields_by_name['value'].message_type = _ALGROBOTACTIONREQ_ACTIONLIST
_ALGROBOTACTIONREQ_ACTIONLISTENTRY.containing_type = _ALGROBOTACTIONREQ
_ALGROBOTACTIONREQ.fields_by_name['action_list'].message_type = _ALGROBOTACTIONREQ_ACTIONLISTENTRY
DESCRIPTOR.message_types_by_name['AlgNodeRegisterReq'] = _ALGNODEREGISTERREQ
DESCRIPTOR.message_types_by_name['AlgNodeRegisterRsp'] = _ALGNODEREGISTERRSP
DESCRIPTOR.message_types_by_name['AlgKeepAliveReq'] = _ALGKEEPALIVEREQ
DESCRIPTOR.message_types_by_name['AlgKeepAliveRsp'] = _ALGKEEPALIVERSP
DESCRIPTOR.message_types_by_name['AlgRobotActionReq'] = _ALGROBOTACTIONREQ
DESCRIPTOR.message_types_by_name['AlgRobotActionRsp'] = _ALGROBOTACTIONRSP
DESCRIPTOR.enum_types_by_name['ALG_CMD'] = _ALG_CMD
DESCRIPTOR.enum_types_by_name['ALG_ERR'] = _ALG_ERR
_sym_db.RegisterFileDescriptor(DESCRIPTOR)

AlgNodeRegisterReq = _reflection.GeneratedProtocolMessageType('AlgNodeRegisterReq', (_message.Message,), dict(
  DESCRIPTOR = _ALGNODEREGISTERREQ,
  __module__ = 'alg_pb2'
  # @@protoc_insertion_point(class_scope:alg.AlgNodeRegisterReq)
  ))
_sym_db.RegisterMessage(AlgNodeRegisterReq)

AlgNodeRegisterRsp = _reflection.GeneratedProtocolMessageType('AlgNodeRegisterRsp', (_message.Message,), dict(
  DESCRIPTOR = _ALGNODEREGISTERRSP,
  __module__ = 'alg_pb2'
  # @@protoc_insertion_point(class_scope:alg.AlgNodeRegisterRsp)
  ))
_sym_db.RegisterMessage(AlgNodeRegisterRsp)

AlgKeepAliveReq = _reflection.GeneratedProtocolMessageType('AlgKeepAliveReq', (_message.Message,), dict(
  DESCRIPTOR = _ALGKEEPALIVEREQ,
  __module__ = 'alg_pb2'
  # @@protoc_insertion_point(class_scope:alg.AlgKeepAliveReq)
  ))
_sym_db.RegisterMessage(AlgKeepAliveReq)

AlgKeepAliveRsp = _reflection.GeneratedProtocolMessageType('AlgKeepAliveRsp', (_message.Message,), dict(
  DESCRIPTOR = _ALGKEEPALIVERSP,
  __module__ = 'alg_pb2'
  # @@protoc_insertion_point(class_scope:alg.AlgKeepAliveRsp)
  ))
_sym_db.RegisterMessage(AlgKeepAliveRsp)

AlgRobotActionReq = _reflection.GeneratedProtocolMessageType('AlgRobotActionReq', (_message.Message,), dict(

  ActionList = _reflection.GeneratedProtocolMessageType('ActionList', (_message.Message,), dict(
    DESCRIPTOR = _ALGROBOTACTIONREQ_ACTIONLIST,
    __module__ = 'alg_pb2'
    # @@protoc_insertion_point(class_scope:alg.AlgRobotActionReq.ActionList)
    ))
  ,

  ActionListEntry = _reflection.GeneratedProtocolMessageType('ActionListEntry', (_message.Message,), dict(
    DESCRIPTOR = _ALGROBOTACTIONREQ_ACTIONLISTENTRY,
    __module__ = 'alg_pb2'
    # @@protoc_insertion_point(class_scope:alg.AlgRobotActionReq.ActionListEntry)
    ))
  ,
  DESCRIPTOR = _ALGROBOTACTIONREQ,
  __module__ = 'alg_pb2'
  # @@protoc_insertion_point(class_scope:alg.AlgRobotActionReq)
  ))
_sym_db.RegisterMessage(AlgRobotActionReq)
_sym_db.RegisterMessage(AlgRobotActionReq.ActionList)
_sym_db.RegisterMessage(AlgRobotActionReq.ActionListEntry)

AlgRobotActionRsp = _reflection.GeneratedProtocolMessageType('AlgRobotActionRsp', (_message.Message,), dict(
  DESCRIPTOR = _ALGROBOTACTIONRSP,
  __module__ = 'alg_pb2'
  # @@protoc_insertion_point(class_scope:alg.AlgRobotActionRsp)
  ))
_sym_db.RegisterMessage(AlgRobotActionRsp)


DESCRIPTOR.has_options = True
DESCRIPTOR._options = _descriptor._ParseOptions(descriptor_pb2.FileOptions(), _b('\200\001\000'))
_ALGROBOTACTIONREQ_ACTIONLISTENTRY.has_options = True
_ALGROBOTACTIONREQ_ACTIONLISTENTRY._options = _descriptor._ParseOptions(descriptor_pb2.MessageOptions(), _b('8\001'))
# @@protoc_insertion_point(module_scope)
