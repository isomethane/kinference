package io.kinference.webgpu.operators.logical

import io.kinference.attribute.Attribute
import io.kinference.data.ONNXData
import io.kinference.graph.Contexts
import io.kinference.ndarray.functions.logical.GreaterOperator
import io.kinference.operator.*
import io.kinference.protobuf.message.TensorProto
import io.kinference.webgpu.data.tensor.WebGPUTensor
import io.kinference.webgpu.data.tensor.asTensor

sealed class Greater(name: String, info: OperatorInfo, attributes: Map<String, Attribute<Any>>, inputs: List<String>, outputs: List<String>)
    : Operator<WebGPUTensor, WebGPUTensor>(name, info, attributes, inputs, outputs) {
    companion object {
        private val DEFAULT_VERSION = VersionInfo(sinceVersion = 7)

        operator fun invoke(name: String, version: Int?, attributes: Map<String, Attribute<Any>>, inputs: List<String>, outputs: List<String>) = when (version ?: DEFAULT_VERSION.sinceVersion) {
            in GreaterVer7.VERSION.asRange() -> GreaterVer7(name, attributes, inputs, outputs)
            else -> error("Unsupported version of Greater operator: $version")
        }
    }
}

class GreaterVer7(name: String, attributes: Map<String, Attribute<Any>>, inputs: List<String>, outputs: List<String>) : Greater(name, INFO, attributes, inputs, outputs) {
    companion object {
        private val TYPE_CONSTRAINTS = PRIMITIVE_DATA_TYPES

        private val INPUTS_INFO = listOf(
            IOInfo(0, TYPE_CONSTRAINTS, "A", optional = false),
            IOInfo(1, TYPE_CONSTRAINTS, "B", optional = false)
        )

        private val OUTPUTS_INFO = listOf(
            IOInfo(0, setOf(TensorProto.DataType.BOOL), "C", optional = false)
        )

        internal val VERSION = VersionInfo(sinceVersion = 7)
        private val INFO = OperatorInfo("Greater", emptyMap(), INPUTS_INFO, OUTPUTS_INFO, VERSION, domain = OperatorInfo.DEFAULT_DOMAIN)
    }

    private val greater = GreaterOperator()

    override suspend fun <D : ONNXData<*, *>> apply(contexts: Contexts<D>, inputs: List<WebGPUTensor?>): List<WebGPUTensor?> {
        return listOf(greater.apply(inputs[0]!!.data, inputs[1]!!.data).asTensor("C"))
    }
}
