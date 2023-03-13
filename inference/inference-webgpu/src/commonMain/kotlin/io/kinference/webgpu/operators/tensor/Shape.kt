package io.kinference.webgpu.operators.tensor

import io.kinference.attribute.Attribute
import io.kinference.data.ONNXData
import io.kinference.graph.Contexts
import io.kinference.ndarray.arrays.*
import io.kinference.operator.IOInfo
import io.kinference.operator.Operator
import io.kinference.operator.OperatorInfo
import io.kinference.operator.VersionInfo
import io.kinference.protobuf.message.TensorProto
import io.kinference.webgpu.data.tensor.WebGPUTensor
import io.kinference.webgpu.data.tensor.asTensor


sealed class Shape(name: String, info: OperatorInfo, attributes: Map<String, Attribute<Any>>, inputs: List<String>, outputs: List<String>) : Operator<WebGPUTensor, WebGPUTensor>(name, info, attributes, inputs, outputs) {
    companion object {
        private val DEFAULT_VERSION = VersionInfo(sinceVersion = 1, untilVersion = 15)

        operator fun invoke(name: String, version: Int?, attributes: Map<String, Attribute<Any>>, inputs: List<String>, outputs: List<String>) = when (version ?: DEFAULT_VERSION.sinceVersion) {
            in ShapeVer1.VERSION.asRange() -> ShapeVer1(name, attributes, inputs, outputs)
            else -> error("Unsupported version of Constant operator: $version")
        }
    }
}

class ShapeVer1(name: String, attributes: Map<String, Attribute<Any>>, inputs: List<String>, outputs: List<String>) : Shape(name, INFO, attributes, inputs, outputs) {
    companion object {
        private val TYPE_CONSTRAINTS = ALL_DATA_TYPES

        private val INPUTS_INFO = listOf(IOInfo(0, TYPE_CONSTRAINTS, "data", optional = false, differentiable = false))

        private val OUTPUTS_INFO = listOf(IOInfo(0, setOf(TensorProto.DataType.INT64), "shape", optional = false, differentiable = false))

        internal val VERSION = VersionInfo(sinceVersion = 1, untilVersion = 15)
        private val INFO = OperatorInfo("Shape", emptyMap(), INPUTS_INFO, OUTPUTS_INFO, VERSION, OperatorInfo.DEFAULT_DOMAIN)
    }


    override suspend fun <D : ONNXData<*, *>> apply(contexts: Contexts<D>, inputs: List<WebGPUTensor?>): List<WebGPUTensor?> {
        val tensor = inputs.first()!!
        val shape = tensor.data.info.shape

        val outputTensorShape = intArrayOf(shape.size)
        val output = NDArrayWebGPU(NDArrayInfo(outputTensorShape, WebGPUDataType.INT32), IntNDArrayData(shape))
        return listOf(output.asTensor("shape"))
    }
}
