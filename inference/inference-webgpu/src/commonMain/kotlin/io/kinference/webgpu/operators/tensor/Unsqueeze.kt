package io.kinference.webgpu.operators.tensor

import io.kinference.attribute.Attribute
import io.kinference.data.ONNXData
import io.kinference.graph.Contexts
import io.kinference.ndarray.extensions.unsqueeze
import io.kinference.ndarray.toIntArray
import io.kinference.operator.*
import kotlin.time.ExperimentalTime
import io.kinference.protobuf.message.AttributeProto
import io.kinference.webgpu.data.tensor.WebGPUTensor
import io.kinference.webgpu.data.tensor.asTensor
import io.kinference.webgpu.engine.WebGPUEnvironment

sealed class Unsqueeze(name: String, info: OperatorInfo, attributes: Map<String, Attribute<Any>>, inputs: List<String>, outputs: List<String>) : Operator<WebGPUTensor, WebGPUTensor>(name, info, attributes, inputs, outputs) {
    companion object {
        private val DEFAULT_VERSION = VersionInfo(sinceVersion = 1, untilVersion = 13)

        operator fun invoke(name: String, version: Int?, attributes: Map<String, Attribute<Any>>, inputs: List<String>, outputs: List<String>) = when (version ?: DEFAULT_VERSION.sinceVersion) {
            in UnsqueezeVer1.VERSION.asRange() -> UnsqueezeVer1(name, attributes, inputs, outputs)
            else -> error("Unsupported version of Unsqueeze operator: $version")
        }
    }
}

@ExperimentalTime
class UnsqueezeVer1(name: String, attributes: Map<String, Attribute<Any>>, inputs: List<String>, outputs: List<String>) : Unsqueeze(name, INFO, attributes, inputs, outputs) {
    companion object {
        private val TYPE_CONSTRAINTS = ALL_DATA_TYPES

        private val ATTRIBUTES_INFO = listOf(
            AttributeInfo("axes", setOf(AttributeProto.AttributeType.INTS), true)
        )

        private val INPUTS_INFO = listOf(IOInfo(0, TYPE_CONSTRAINTS, "data", optional = false))

        private val OUTPUTS_INFO = listOf(IOInfo(0, TYPE_CONSTRAINTS, "expanded", optional = false))

        internal val VERSION = VersionInfo(sinceVersion = 1, untilVersion = 13)
        private val INFO = OperatorInfo("Unsqueeze", ATTRIBUTES_INFO, INPUTS_INFO, OUTPUTS_INFO, VERSION, OperatorInfo.DEFAULT_DOMAIN)
    }

    private val axes: IntArray by attribute { it: LongArray -> it.toIntArray() }

    override suspend fun <D : ONNXData<*, *>> apply(contexts: Contexts<D>, inputs: List<WebGPUTensor?>): List<WebGPUTensor?> {
        return listOf(inputs[0]!!.data.unsqueeze(axes, WebGPUEnvironment.gpuState).asTensor())
    }
}
