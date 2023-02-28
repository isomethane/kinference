package io.kinference.protobuf

import com.squareup.wire.FieldEncoding
import com.squareup.wire.ProtoAdapter
import io.kinference.protobuf.arrays.ArrayContainer
import io.kinference.protobuf.message.StringStringEntryProto
import io.kinference.protobuf.message.TensorProto
import io.kinference.protobuf.message.TensorProto.Companion.hasData
import okio.Buffer
import okio.ByteString

abstract class TensorDecoder {
    protected abstract fun initContainer(): ArrayContainer
    protected abstract fun makeArray(type: TensorProto.DataType, shape: IntArray, init: (Int) -> Any): Any
    protected abstract fun parseInt32Data(proto: TensorProto): Any
    protected abstract fun hasIntArray(proto: TensorProto): Boolean

    private fun TensorProto.checkArrayData() {
        if (!hasIntArray(this)) return
        if (dataType == TensorProto.DataType.INT32) return

        require(dataType in int32AvailableTypes) { "Conversion from int32 to $dataType is not supported" }
        val newArray = parseInt32Data(this)
        _arrayData!!.setData(newArray)
    }

    fun decode(reader: ProtobufReader): TensorProto {
        val proto = TensorProto(_arrayData = initContainer())
        var rawData: ByteString? = null
        reader.forEachTag { tag ->
            when (TensorProto.ReaderTag.fromInt(tag)) {
                TensorProto.ReaderTag.DIMS -> proto.dims = reader.readLongArray(tag).toIntArray()
                TensorProto.ReaderTag.DATATYPE -> proto.dataType = reader.readValue(TensorProto.DataType.ADAPTER)
                TensorProto.ReaderTag.SEGMENT -> proto.segment = TensorProto.Segment.decode(reader)
                TensorProto.ReaderTag.FLOAT -> reader.readFloatArray(tag, proto.dims, proto._arrayData!!)
                TensorProto.ReaderTag.INT32 -> reader.readIntArray(tag, proto.dims, proto._arrayData!!)
                TensorProto.ReaderTag.STRING -> proto.stringData.add(reader.readBytes())
                TensorProto.ReaderTag.INT64 -> reader.readLongArray(tag, proto.dims, proto._arrayData!!)
                TensorProto.ReaderTag.NAME -> proto.name = reader.readString()
                TensorProto.ReaderTag.RAW -> rawData = reader.readBytes()
                TensorProto.ReaderTag.DOUBLE -> reader.readDoubleArray(tag, proto.dims, proto._arrayData!!)
                TensorProto.ReaderTag.UINT64 -> reader.readULongArray(tag, proto.dims, proto._arrayData!!)
                TensorProto.ReaderTag.DOC_STRING -> reader.readString() // skip docstring
                TensorProto.ReaderTag.EXTERNAL -> proto.externalData.add(StringStringEntryProto.decode(reader))
                TensorProto.ReaderTag.LOCATION -> try {
                    proto.dataLocation = reader.readValue(TensorProto.DataLocation.ADAPTER)
                } catch (e: ProtoAdapter.EnumConstantNotFoundException) {
                    reader.addUnknownField(tag, FieldEncoding.VARINT, e.value.toLong())
                }
                null -> reader.readUnknownField(tag)
            }
        }
        if (rawData != null || !proto.hasData()) parseRaw(rawData, proto)
        proto.checkArrayData()
        return proto
    }

    private fun parseRaw(rawData: ByteString?, proto: TensorProto) {
        require(proto._arrayData != null)
        val raw = rawData ?: ByteString.EMPTY
        val buffer = Buffer().apply { write(raw) }
        val shape = proto.dims

        when (proto.dataType) {
            TensorProto.DataType.DOUBLE -> proto._arrayData!!.setData(makeArray(proto.dataType!!, shape) { buffer.readDoubleLe() })
            TensorProto.DataType.FLOAT -> proto._arrayData!!.setData(makeArray(proto.dataType!!, shape) { buffer.readFloatLe() })
            TensorProto.DataType.INT64 -> proto._arrayData!!.setData(makeArray(proto.dataType!!, shape) { buffer.readLongLe() })
            TensorProto.DataType.INT32 -> proto._arrayData!!.setData(makeArray(proto.dataType!!, shape) { buffer.readIntLe() })
            TensorProto.DataType.INT16 -> proto._arrayData!!.setData(makeArray(proto.dataType!!, shape) { buffer.readShortLe() })
            TensorProto.DataType.UINT16 -> proto._arrayData!!.setData(makeArray(proto.dataType!!, shape) { buffer.readShortLe().toUShort() })
            TensorProto.DataType.INT8 -> proto._arrayData!!.setData(makeArray(proto.dataType!!, shape) { buffer.readByte() })
            TensorProto.DataType.UINT8 -> proto._arrayData!!.setData(makeArray(proto.dataType!!, shape) { buffer.readByte().toUByte() })
            TensorProto.DataType.BOOL -> proto._arrayData!!.setData(makeArray(proto.dataType!!, shape) { buffer.readByte() != 0.toByte() })
            TensorProto.DataType.STRING -> error("String data must not be present in rawData field")
            else -> error("Unsupported data type ${proto.dataType}")
        }
    }

    companion object {
        private val int32AvailableTypes = setOf(TensorProto.DataType.BOOL, TensorProto.DataType.INT8, TensorProto.DataType.UINT8, TensorProto.DataType.INT16, TensorProto.DataType.UINT16)
    }
}
