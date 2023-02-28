package io.kinference.webgpu.data.tensor

import io.kinference.ndarray.arrays.NDArrayWebGPU
import io.kinference.ndarray.arrays.WebGPUDataType
import io.kinference.protobuf.message.TensorProto
import io.kinference.types.TensorShape
import io.kinference.types.ValueTypeInfo

fun NDArrayWebGPU.asTensor(name: String? = null) =
    WebGPUTensor(name,this, ValueTypeInfo.TensorTypeInfo(TensorShape(info.shape), info.type.resolve()))

private fun WebGPUDataType.resolve(): TensorProto.DataType = when (this) {
    WebGPUDataType.INT32 -> TensorProto.DataType.INT32
    WebGPUDataType.UINT32 -> TensorProto.DataType.UINT32
    WebGPUDataType.FLOAT32 -> TensorProto.DataType.FLOAT
}
