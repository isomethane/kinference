package io.kinference.webgpu.operators.math

import io.kinference.attribute.Attribute
import io.kinference.data.ONNXData
import io.kinference.graph.Contexts
import io.kinference.ndarray.functions.math.DivOperator
import io.kinference.operator.*
import io.kinference.protobuf.message.TensorProto
import io.kinference.webgpu.data.tensor.WebGPUTensor
import io.kinference.webgpu.data.tensor.asTensor

sealed class Div(name: String, info: OperatorInfo, attributes: Map<String, Attribute<Any>>, inputs: List<String>, outputs: List<String>)
    : Operator<WebGPUTensor, WebGPUTensor>(name, info, attributes, inputs, outputs) {
    companion object {
        private val DEFAULT_VERSION = VersionInfo(sinceVersion = 7)

        operator fun invoke(name: String, version: Int?, attributes: Map<String, Attribute<Any>>, inputs: List<String>, outputs: List<String>) = when (version ?: DEFAULT_VERSION.sinceVersion) {
            in DivVer7.VERSION.asRange() -> DivVer7(name, attributes, inputs, outputs)
            else -> error("Unsupported version of Div operator: $version")
        }
    }
}

class DivVer7(name: String, attributes: Map<String, Attribute<Any>>, inputs: List<String>, outputs: List<String>) : Div(name, INFO, attributes, inputs, outputs) {
    companion object {
        private val TYPE_CONSTRAINTS = setOf(
            TensorProto.DataType.UINT32,
            TensorProto.DataType.UINT64,
            TensorProto.DataType.INT32,
            TensorProto.DataType.INT64,
            TensorProto.DataType.FLOAT,
            TensorProto.DataType.DOUBLE,
        )

        private val INPUTS_INFO = listOf(
            IOInfo(0, TYPE_CONSTRAINTS, "A", optional = false),
            IOInfo(1, TYPE_CONSTRAINTS, "B", optional = false)
        )

        private val OUTPUTS_INFO = listOf(
            IOInfo(0, TYPE_CONSTRAINTS, "C", optional = false)
        )

        internal val VERSION = VersionInfo(sinceVersion = 7)
        private val INFO = OperatorInfo("Div", emptyMap(), INPUTS_INFO, OUTPUTS_INFO, VERSION, domain = OperatorInfo.DEFAULT_DOMAIN)
    }

    private val div = DivOperator()

    override suspend fun <D : ONNXData<*, *>> apply(contexts: Contexts<D>, inputs: List<WebGPUTensor?>): List<WebGPUTensor?> {
        return listOf(div.apply(inputs[0]!!.data, inputs[1]!!.data).asTensor("C"))
    }
}
