package io.kinference.webgpu.operators.tensor

import io.kinference.attribute.Attribute
import io.kinference.data.ONNXData
import io.kinference.graph.Contexts
import io.kinference.ndarray.arrays.*
import io.kinference.operator.*
import io.kinference.protobuf.message.AttributeProto
import io.kinference.protobuf.message.TensorProto
import io.kinference.webgpu.data.tensor.WebGPUTensor
import io.kinference.webgpu.data.tensor.asTensor
import io.kinference.webgpu.engine.WebGPUEnvironment

sealed class ConstantOfShape(name: String, info: OperatorInfo, attributes: Map<String, Attribute<Any>>, inputs: List<String>, outputs: List<String>) : Operator<WebGPUTensor, WebGPUTensor>(name, info, attributes, inputs, outputs) {
    companion object {
        private val DEFAULT_VERSION = VersionInfo(sinceVersion = 9)

        operator fun invoke(name: String, version: Int?, attributes: Map<String, Attribute<Any>>, inputs: List<String>, outputs: List<String>) = when (version ?: DEFAULT_VERSION.sinceVersion) {
            in ConstantOfShapeVer9.VERSION.asRange() -> ConstantOfShapeVer9(name, attributes, inputs, outputs)
            else -> error("Unsupported version of ConstantOfShape operator: $version")
        }
    }
}

class ConstantOfShapeVer9(name: String, attributes: Map<String, Attribute<Any>>, inputs: List<String>, outputs: List<String>) : ConstantOfShape(name, INFO, attributes, inputs, outputs) {
    companion object {
        private val TYPE_CONSTRAINTS = PRIMITIVE_DATA_TYPES

        private val DEFAULT_TENSOR = NDArrayWebGPU.scalar(0f).asTensor("value")
        private val ATTRIBUTES_INFO = listOf(
            AttributeInfo("value", setOf(AttributeProto.AttributeType.TENSOR), default = DEFAULT_TENSOR, required = false)
        )

        private val INPUTS_INFO = listOf(IOInfo(0, setOf(TensorProto.DataType.INT64), "input", optional = false))

        private val OUTPUTS_INFO = listOf(IOInfo(0, TYPE_CONSTRAINTS, "output", optional = false))

        internal val VERSION = VersionInfo(sinceVersion = 9)
        private val INFO = OperatorInfo("ConstantOfShape", ATTRIBUTES_INFO, INPUTS_INFO, OUTPUTS_INFO, VERSION, OperatorInfo.DEFAULT_DOMAIN)
    }

    private val value: WebGPUTensor by attribute()

    override fun <D : ONNXData<*, *>> apply(contexts: Contexts<D>, inputs: List<WebGPUTensor?>): List<WebGPUTensor?> {
        error("Use applySuspend()")
    }

    override suspend fun <D : ONNXData<*, *>> applySuspend(contexts: Contexts<D>, inputs: List<WebGPUTensor?>): List<WebGPUTensor?> {
        val shape = (inputs[0]!!.data.getData(WebGPUEnvironment.gpuState) as IntNDArrayData).data
        val size = shape.fold(1, Int::times)
        val data: TypedNDArrayData = when (val ndArrayData = value.data.getData()) {
            is IntNDArrayData -> IntNDArrayData(IntArray(size).apply { fill(ndArrayData.data[0]) })
            is UIntNDArrayData -> UIntNDArrayData(UIntArray(size).apply { fill(ndArrayData.data[0]) })
            is FloatNDArrayData -> FloatNDArrayData(FloatArray(size).apply { fill(ndArrayData.data[0]) })
        }
        val result = NDArrayWebGPU(NDArrayInfo(shape, value.data.info.type), data)
        return listOf(result.asTensor("output"))
    }
}
