package io.kinference.tfjs.operators.layer.normalization

import io.kinference.attribute.Attribute
import io.kinference.data.ONNXData
import io.kinference.graph.Contexts
import io.kinference.operator.*
import io.kinference.protobuf.message.AttributeProto
import io.kinference.protobuf.message.TensorProto
import io.kinference.tfjs.data.tensors.TFJSTensor
import io.kinference.tfjs.data.tensors.asTensor
import io.kinference.tfjs.externals.core.scalar
import io.kinference.tfjs.externals.extensions.*

sealed class SkipLayerNormalization(info: OperatorInfo, attributes: Map<String, Attribute<Any>>, inputs: List<String>, outputs: List<String>)
    : Operator<TFJSTensor, TFJSTensor>(info, attributes, inputs, outputs) {
    companion object {
        private val DEFAULT_VERSION = VersionInfo(sinceVersion = 1)

        operator fun invoke(version: Int?, attributes: Map<String, Attribute<Any>>, inputs: List<String>, outputs: List<String>) = when (version ?: DEFAULT_VERSION.sinceVersion) {
            in SkipLayerNormalizationVer1.VERSION.asRange() -> SkipLayerNormalizationVer1(attributes, inputs, outputs)
            else -> error("Unsupported version of SkipLayerNormalization operator: $version")
        }
    }
}

class SkipLayerNormalizationVer1(attributes: Map<String, Attribute<Any>>, inputs: List<String>, outputs: List<String>) : SkipLayerNormalization(INFO, attributes, inputs, outputs) {

    companion object {
        private val TYPE_CONSTRAINTS = setOf(
            TensorProto.DataType.FLOAT,
            TensorProto.DataType.FLOAT16
        )

        private val ATTRIBUTES_INFO = listOf(
            AttributeInfo("epsilon", setOf(AttributeProto.AttributeType.FLOAT), false, 0.00001f)
        )

        private val INPUTS_INFO = listOf(
            IOInfo(0, TYPE_CONSTRAINTS, "input", false),
            IOInfo(1, TYPE_CONSTRAINTS, "skip", false),
            IOInfo(2, TYPE_CONSTRAINTS, "gamma", false),
            IOInfo(3, TYPE_CONSTRAINTS, "beta", false),
            IOInfo(4, TYPE_CONSTRAINTS, "bias", true)
        )

        private val OUTPUTS_INFO = listOf(
            IOInfo(0, TYPE_CONSTRAINTS, "output", false),
            //Only for training, not supported now
            IOInfo(1, TYPE_CONSTRAINTS, "mean", true),
            IOInfo(2, TYPE_CONSTRAINTS, "inv_std_var", true)
        )

        internal val VERSION = VersionInfo(sinceVersion = 1)
        private val INFO = OperatorInfo("SkipLayerNormalization", ATTRIBUTES_INFO, INPUTS_INFO, OUTPUTS_INFO, VERSION, domain = "com.microsoft")
    }

    private val epsilon: Float by attribute()

    override fun <D : ONNXData<*, *>> apply(contexts: Contexts<D>, inputs: List<TFJSTensor?>): List<TFJSTensor?> {
        val outputs = tidy {
            val input = inputs[0]!!.data
            val skip = inputs[1]!!.data
            val gamma = inputs[2]!!.data
            val beta = inputs[3]!!.data
            val bias = inputs.getOrNull(4)?.data

            val skippedInput = if (bias != null) {
                input.add(skip, bias.broadcastTo(input.shape))
            } else {
                input.plus(skip)
            }

            val momentsOutput = skippedInput.moments(-1, true)
            val mean = momentsOutput.mean
            val variance = momentsOutput.variance

            val epsilonTensor = scalar(epsilon, "float32")
            val output = (skippedInput - mean) / (sqrt(variance + epsilonTensor)) * gamma + beta
            return@tidy arrayOf(output)
        }

        return listOf(outputs[0].asTensor("output"))
    }
}
