package io.kinference.tfjs.operators

import io.kinference.protobuf.message.NodeProto
import io.kinference.tfjs.TFJSData
import io.kinference.tfjs.attributes.Attribute
import io.kinference.tfjs.model.TFJSModel
import io.kinference.tfjs.operators.layer.attention.Attention
import io.kinference.tfjs.operators.layer.attention.QAttention
import io.kinference.tfjs.operators.layer.normalization.*
import io.kinference.tfjs.operators.math.*
import io.kinference.tfjs.operators.quantization.DequantizeLinear
import io.kinference.tfjs.operators.quantization.DynamicQuantizeLinear
import io.kinference.tfjs.operators.tensor.*

object OperatorFactory {
    @Suppress("UNCHECKED_CAST")
    fun create(opType: String?, version: Int?, attributes: Map<String, Attribute<Any>>, inputs: List<String>, outputs: List<String>) = when (opType) {
        "Attention" -> Attention(version, attributes, inputs, outputs)
        "Add" -> Add(version, attributes, inputs, outputs)
        "Shape" -> Shape(version, attributes, inputs, outputs)
        "ConstantOfShape" -> ConstantOfShape(version, attributes, inputs, outputs)
        "Cast" -> Cast(version, attributes, inputs, outputs)
        "EmbedLayerNormalization" -> EmbedLayerNormalization(version, attributes, inputs, outputs)
        "MatMul" -> MatMul(version, attributes, inputs, outputs)
        "SkipLayerNormalization" -> SkipLayerNormalization(version, attributes, inputs, outputs)
        "BiasGelu" -> BiasGelu(version, attributes, inputs, outputs)
        "DequantizeLinear" -> DequantizeLinear(version, attributes, inputs, outputs)
        "DynamicQuantizeLinear" -> DynamicQuantizeLinear(version, attributes, inputs, outputs)
        "LayerNormalization" -> LayerNormalization(version, attributes, inputs, outputs)
        "QAttention" -> QAttention(version, attributes, inputs, outputs)
        "Gather" -> Gather(version, attributes, inputs, outputs)
        "Unsqueeze" -> Unsqueeze(version, attributes, inputs, outputs)
        "Concat" -> Concat(version, attributes, inputs, outputs)
        "Reshape" -> Reshape(version, attributes, inputs, outputs)
        "Mul" -> Mul(version, attributes, inputs, outputs)
        "FastGelu" -> FastGelu(version, attributes, inputs, outputs)
        "MatMulInteger" -> MatMulInteger(version, attributes, inputs, outputs)
        "Slice" -> Slice(version, attributes, inputs, outputs)
        "Squeeze" -> Squeeze(version, attributes, inputs, outputs)
        else -> error("Unsupported operator: $opType")
    } as Operator<TFJSData<*>, TFJSData<*>>

    fun create(proto: NodeProto, opSetRegistry: TFJSModel.OperatorSetRegistry): Operator<TFJSData<*>, TFJSData<*>> {
        val version = opSetRegistry.getVersion(proto.domain)
        return create(proto.opType, version, proto.attribute.map { Attribute.create(it, opSetRegistry) }.associateBy { it.name }, proto.input, proto.output)
    }
}
