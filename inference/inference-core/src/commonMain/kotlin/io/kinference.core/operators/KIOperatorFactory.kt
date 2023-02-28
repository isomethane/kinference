package io.kinference.core.operators

import io.kinference.attribute.Attribute
import io.kinference.attribute.AttributeFactory
import io.kinference.core.KIONNXData
import io.kinference.core.data.tensor.KITensor
import io.kinference.core.graph.KIGraph
import io.kinference.core.operators.activations.*
import io.kinference.core.operators.flow.*
import io.kinference.core.operators.layer.attention.Attention
import io.kinference.core.operators.layer.attention.QAttention
import io.kinference.core.operators.layer.normalization.*
import io.kinference.core.operators.layer.recurrent.gru.GRU
import io.kinference.core.operators.layer.recurrent.lstm.LSTM
import io.kinference.core.operators.logical.*
import io.kinference.core.operators.math.*
import io.kinference.core.operators.ml.*
import io.kinference.core.operators.quantization.*
import io.kinference.core.operators.seq.ConcatFromSequence
import io.kinference.core.operators.seq.SplitToSequence
import io.kinference.core.operators.tensor.*
import io.kinference.core.operators.quantization.lstm.DynamicQuantizeLSTM
import io.kinference.graph.Graph
import io.kinference.operator.*
import io.kinference.protobuf.message.*
import kotlin.time.ExperimentalTime

object KIAttributeFactory : AttributeFactory<KIONNXData<*>> {
    override fun createTensor(proto: TensorProto): KIONNXData<*> = KITensor.create(proto)
    override fun createGraph(proto: GraphProto, opSet: OperatorSetRegistry): Graph<KIONNXData<*>> = KIGraph(proto, opSet)
}

@ExperimentalTime
object KIOperatorFactory : OperatorFactory<KIONNXData<*>> {
    override fun attributeFactory(): AttributeFactory<KIONNXData<*>> = KIAttributeFactory

    @Suppress("UNCHECKED_CAST")
    override fun create(name: String, opType: String?, version: Int?, attributes: Map<String, Attribute<Any>>, inputs: List<String>, outputs: List<String>) = when (opType) {
        "Add" -> Add(name, version, attributes, inputs, outputs)
        "ArgMax" -> ArgMax(name, version, attributes, inputs, outputs)
        "Sub" -> Sub(name, version, attributes, inputs, outputs)
        "Attention" -> Attention(name, version, attributes, inputs, outputs)
        "BiasGelu" -> BiasGelu(name, version, attributes, inputs, outputs)
        "Cast" -> Cast(name, version, attributes, inputs, outputs)
        "Concat" -> Concat(name, version, attributes, inputs, outputs)
        "ConcatFromSequence" -> ConcatFromSequence(name, version, attributes, inputs, outputs)
        "Constant" -> Constant(name, version, attributes, inputs, outputs)
        "ConstantOfShape" -> ConstantOfShape(name, version, attributes, inputs, outputs)
        "CumSum" -> CumSum(name, version, attributes, inputs, outputs)
        "DequantizeLinear" -> DequantizeLinear(name, version, attributes, inputs, outputs)
        "Div" -> Div(name, version, attributes, inputs, outputs)
        "DynamicQuantizeLinear" -> DynamicQuantizeLinear(name, version, attributes, inputs, outputs)
        "DynamicQuantizeLSTM" -> DynamicQuantizeLSTM(name, version, attributes, inputs, outputs)
        "DynamicQuantizeMatMul" -> DynamicQuantizeMatMul(name, version, attributes, inputs, outputs)
        "EmbedLayerNormalization" -> EmbedLayerNormalization(name, version, attributes, inputs, outputs)
        "Equal" -> Equal(name, version, attributes, inputs, outputs)
        "Erf" -> Erf(name, version, attributes, inputs, outputs)
        "Expand" -> Expand(name, version, attributes, inputs, outputs)
        "ReduceSum" -> ReduceSum(name, version, attributes, inputs, outputs)
        "FastGelu" -> FastGelu(name, version, attributes, inputs, outputs)
        "Flatten" -> Flatten(name, version, attributes, inputs, outputs)
        "FusedMatMul" -> FusedMatMul(name, version, attributes, inputs, outputs)
        "Gather" -> Gather(name, version, attributes, inputs, outputs)
        "GatherElements" -> GatherElements(name, version, attributes, inputs, outputs)
        "GatherND" -> GatherND(name, version, attributes, inputs, outputs)
        "Gelu" -> Gelu(name, version, attributes, inputs, outputs)
        "Gemm" -> Gemm(name, version, attributes, inputs, outputs)
        "Greater" -> Greater(name, version, attributes, inputs, outputs)
        "GRU" -> GRU(name, version, attributes, inputs, outputs)
        "Identity" -> Identity(name, version, attributes, inputs, outputs)
        "If" -> If(name, version, attributes, inputs, outputs)
        "Less" -> Less(name, version, attributes, inputs, outputs)
        "LSTM" -> LSTM(name, version, attributes, inputs, outputs)
        "Loop" -> Loop(name, version, attributes, inputs, outputs)
        "LayerNormalization" -> LayerNormalization(name, version, attributes, inputs, outputs)
        "LeakyRelu" -> LeakyRelu(name, version, attributes, inputs, outputs)
        "Log" -> Log(name, version, attributes, inputs, outputs)
        "LogSoftmax" -> LogSoftmax(name, version, attributes, inputs, outputs)
        "MatMul" -> MatMul(name, version, attributes, inputs, outputs)
        "MatMulInteger" -> MatMulInteger(name, version, attributes, inputs, outputs)
        "MatMulIntegerToFloat" -> MatMulIntegerToFloat(name, version, attributes, inputs, outputs)
        "Mul" -> Mul(name, version, attributes, inputs, outputs)
        "NonZero" -> NonZero(name, version, attributes, inputs, outputs)
        "Not" -> Not(name, version, attributes, inputs, outputs)
        "Or" -> Or(name, version, attributes, inputs, outputs)
        "QAttention" -> QAttention(name, version, attributes, inputs, outputs)
        "QEmbedLayerNormalization" -> QEmbedLayerNormalization(name, version, attributes, inputs, outputs)
        "Range" -> Range(name, version, attributes, inputs, outputs)
        "Relu" -> Relu(name, version, attributes, inputs, outputs)
        "Reshape" -> Reshape(name, version, attributes, inputs, outputs)
        "ScatterElements" -> ScatterElements(name, version, attributes, inputs, outputs)
        "ScatterND" -> ScatterND(name, version, attributes, inputs, outputs)
        "Shape" -> Shape(name, version, attributes, inputs, outputs)
        "Sigmoid" -> Sigmoid(name, version, attributes, inputs, outputs)
        "SkipLayerNormalization" -> SkipLayerNormalization(name, version, attributes, inputs, outputs)
        "Slice" -> Slice(name, version, attributes, inputs, outputs)
        "Softmax" -> Softmax(name, version, attributes, inputs, outputs)
        "Split" -> Split(name, version, attributes, inputs, outputs)
        "SplitToSequence" -> SplitToSequence(name, version, attributes, inputs, outputs)
        "Squeeze" -> Squeeze(name, version, attributes, inputs, outputs)
        "Tanh" -> Tanh(name, version, attributes, inputs, outputs)
        "TopK" -> TopK(name, version, attributes, inputs, outputs)
        "Transpose" -> Transpose(name, version, attributes, inputs, outputs)
        "TreeEnsembleClassifier" -> TreeEnsembleClassifier(name, version, attributes, inputs, outputs)
        "TreeEnsembleRegressor" -> TreeEnsembleRegressor(name, version, attributes, inputs, outputs)
        "Tile" -> Tile(name, version, attributes, inputs, outputs)
        "Pad" -> Pad(name, version, attributes, inputs, outputs)
        "Unsqueeze" -> Unsqueeze(name, version, attributes, inputs, outputs)
        "Where" -> Where(name, version, attributes, inputs, outputs)
        "ZipMap" -> ZipMap(name, version, attributes, inputs, outputs)
        else -> error("Unsupported operator: $opType")
    } as Operator<KIONNXData<*>, KIONNXData<*>>
}
