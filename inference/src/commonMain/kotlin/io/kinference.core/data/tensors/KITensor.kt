package io.kinference.core.data.tensors

import io.kinference.core.data.KIONNXData
import io.kinference.data.ONNXDataType
import io.kinference.ndarray.Strides
import io.kinference.ndarray.arrays.*
import io.kinference.ndarray.arrays.tiled.*
import io.kinference.protobuf.message.TensorProto
import io.kinference.protobuf.message.TensorProto.DataType
import io.kinference.core.types.ValueInfo
import io.kinference.ndarray.extensions.createArray

//TODO: support segments
//TODO: support external data
class KITensor(data: NDArray, info: ValueInfo) : KIONNXData<NDArray>(ONNXDataType.ONNX_TENSOR, data, info) {
    override fun rename(name: String): KITensor {
        return KITensor(data, ValueInfo(info.typeInfo, name))
    }

    companion object {
        //TODO: complex, uint32/64 tensors
        @Suppress("UNCHECKED_CAST")
        fun create(proto: TensorProto): KITensor {
            val type = proto.dataType ?: DataType.UNDEFINED
            val array = parseArray(proto)
            requireNotNull(array) { "Array value should be initialized" }

            return KITensor(array, type, proto.dims, proto.name)
        }

        private operator fun invoke(value: Any, type: DataType, dims: IntArray = IntArray(0), name: String? = ""): KITensor {
            val name = name ?: ""
            val strides = Strides(dims)
            return when (type) {
                DataType.DOUBLE -> DoubleNDArray(value as DoubleTiledArray, strides).asTensor(name)
                DataType.FLOAT -> FloatNDArray(value as FloatTiledArray, strides).asTensor(name)
                DataType.INT32 -> IntNDArray(value as IntTiledArray, strides).asTensor(name)
                DataType.INT8 -> ByteNDArray(value as ByteTiledArray, strides).asTensor(name)
                DataType.UINT8 -> UByteNDArray(value as UByteTiledArray, strides).asTensor(name)
                DataType.INT64 -> LongNDArray(value as LongTiledArray, strides).asTensor(name)
                DataType.INT16 -> ShortNDArray(value as ShortTiledArray, strides).asTensor(name)
                DataType.BOOL -> BooleanNDArray(value as BooleanTiledArray, strides).asTensor(name)
                DataType.STRING -> {
                    value as List<String>
                    StringNDArray(dims) { value[it] }.asTensor(name)
                }
                else -> error("Unsupported data type $type")
            }
        }

        private fun parseArray(proto: TensorProto) = when {
            proto.isTiled() -> proto.arrayData
            proto.isString() -> proto.stringData
            proto.isPrimitive() -> createArray(proto.dims, proto.arrayData!!)
            else -> error("Unsupported data type ${proto.dataType}")
        }
    }
}
