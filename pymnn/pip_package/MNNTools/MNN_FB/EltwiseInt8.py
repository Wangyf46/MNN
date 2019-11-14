# automatically generated by the FlatBuffers compiler, do not modify

# namespace: MNN

import flatbuffers

class EltwiseInt8(object):
    __slots__ = ['_tab']

    @classmethod
    def GetRootAsEltwiseInt8(cls, buf, offset):
        n = flatbuffers.encode.Get(flatbuffers.packer.uoffset, buf, offset)
        x = EltwiseInt8()
        x.Init(buf, n + offset)
        return x

    # EltwiseInt8
    def Init(self, buf, pos):
        self._tab = flatbuffers.table.Table(buf, pos)

    # EltwiseInt8
    def Type(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(4))
        if o != 0:
            return self._tab.Get(flatbuffers.number_types.Int8Flags, o + self._tab.Pos)
        return 0

    # EltwiseInt8
    def InputQuan0(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(6))
        if o != 0:
            x = self._tab.Indirect(o + self._tab.Pos)
            from .QuantizedFloatParam import QuantizedFloatParam
            obj = QuantizedFloatParam()
            obj.Init(self._tab.Bytes, x)
            return obj
        return None

    # EltwiseInt8
    def InputQuan1(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(8))
        if o != 0:
            x = self._tab.Indirect(o + self._tab.Pos)
            from .QuantizedFloatParam import QuantizedFloatParam
            obj = QuantizedFloatParam()
            obj.Init(self._tab.Bytes, x)
            return obj
        return None

    # EltwiseInt8
    def OutputQuan(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(10))
        if o != 0:
            x = self._tab.Indirect(o + self._tab.Pos)
            from .QuantizedFloatParam import QuantizedFloatParam
            obj = QuantizedFloatParam()
            obj.Init(self._tab.Bytes, x)
            return obj
        return None

def EltwiseInt8Start(builder): builder.StartObject(4)
def EltwiseInt8AddType(builder, type): builder.PrependInt8Slot(0, type, 0)
def EltwiseInt8AddInputQuan0(builder, inputQuan0): builder.PrependUOffsetTRelativeSlot(1, flatbuffers.number_types.UOffsetTFlags.py_type(inputQuan0), 0)
def EltwiseInt8AddInputQuan1(builder, inputQuan1): builder.PrependUOffsetTRelativeSlot(2, flatbuffers.number_types.UOffsetTFlags.py_type(inputQuan1), 0)
def EltwiseInt8AddOutputQuan(builder, outputQuan): builder.PrependUOffsetTRelativeSlot(3, flatbuffers.number_types.UOffsetTFlags.py_type(outputQuan), 0)
def EltwiseInt8End(builder): return builder.EndObject()
