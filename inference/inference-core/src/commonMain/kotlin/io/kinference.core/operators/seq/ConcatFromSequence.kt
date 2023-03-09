package io.kinference.core.operators.seq

import io.kinference.attribute.Attribute
import io.kinference.data.ONNXDataType
import io.kinference.core.data.seq.KIONNXSequence
import io.kinference.core.data.tensor.*
import io.kinference.data.ONNXData
import io.kinference.graph.Contexts
import io.kinference.operator.*
import kotlin.time.ExperimentalTime
import io.kinference.protobuf.message.AttributeProto

sealed class ConcatFromSequence(name: String, info: OperatorInfo, attributes: Map<String, Attribute<Any>>, inputs: List<String>, outputs: List<String>) : Operator<KIONNXSequence, KITensor>(name, info, attributes, inputs, outputs) {
    companion object {
        private val DEFAULT_VERSION = VersionInfo(sinceVersion = 11)

        operator fun invoke(name: String, version: Int?, attributes: Map<String, Attribute<Any>>, inputs: List<String>, outputs: List<String>) = when (version ?: DEFAULT_VERSION.sinceVersion) {
            in ConcatFromSequenceVer11.VERSION.asRange() -> ConcatFromSequenceVer11(name, attributes, inputs, outputs)
            else -> error("Unsupported version of ConcatFromSequence operator: $version")
        }
    }
}

@ExperimentalTime
class ConcatFromSequenceVer11(name: String, attributes: Map<String, Attribute<Any>>, inputs: List<String>, outputs: List<String>) : ConcatFromSequence(name, INFO, attributes, inputs, outputs) {
    companion object {
        private val TYPE_CONSTRAINTS = ALL_DATA_TYPES

        private val ATTRIBUTES_INFO = listOf(
            AttributeInfo("axis", setOf(AttributeProto.AttributeType.INT), true),
            AttributeInfo("new_axis", setOf(AttributeProto.AttributeType.INT), false, default = 0L)
        )

        private val INPUTS_INFO = listOf(IOInfo(0, TYPE_CONSTRAINTS, "input_sequence", optional = false, onnxDataType = ONNXDataType.ONNX_SEQUENCE))

        private val OUTPUTS_INFO = listOf(IOInfo(0, TYPE_CONSTRAINTS, "concat_result", optional = false))

        internal val VERSION = VersionInfo(sinceVersion = 11)
        private val INFO = OperatorInfo("ConcatFromSequence", ATTRIBUTES_INFO, INPUTS_INFO, OUTPUTS_INFO, VERSION, OperatorInfo.DEFAULT_DOMAIN)
    }

    private val axis: Int by attribute { it: Number -> it.toInt() }
    private val newAxis: Boolean by attribute("new_axis") { it: Number -> it.toInt() == 1 }

    override suspend fun <D : ONNXData<*, *>> apply(contexts: Contexts<D>, inputs: List<KIONNXSequence?>): List<KITensor?> {
        val srcTensors = inputs.first()!!.data as List<KITensor>
        val tensor = if (newAxis) srcTensors.stack(axis) else srcTensors.concatenate(axis)
        return listOf(tensor)
    }
}
