package io.kinference.webgpu.operators

import io.kinference.attribute.Attribute
import io.kinference.attribute.AttributeFactory
import io.kinference.graph.Graph
import io.kinference.operator.*
import io.kinference.protobuf.message.GraphProto
import io.kinference.protobuf.message.TensorProto
import io.kinference.webgpu.data.tensor.WebGPUTensor
import io.kinference.webgpu.engine.WebGPUData
import io.kinference.webgpu.graph.WebGPUGraph
import io.kinference.webgpu.operators.logical.*
import io.kinference.webgpu.operators.math.*
import io.kinference.webgpu.operators.tensor.*

object WebGPUAttributeFactory : AttributeFactory<WebGPUData<*>> {
    override fun createTensor(proto: TensorProto): WebGPUData<*> = WebGPUTensor.create(proto)
    override suspend fun createGraph(proto: GraphProto, opSet: OperatorSetRegistry): Graph<WebGPUData<*>> = WebGPUGraph(proto, opSet)
}

object WebGPUOperatorFactory : OperatorFactory<WebGPUData<*>> {
    override fun attributeFactory() = WebGPUAttributeFactory

    @Suppress("UNCHECKED_CAST")
    override fun create(
        name: String,
        opType: String?,
        version: Int?,
        attributes: Map<String, Attribute<Any>>,
        inputs: List<String>,
        outputs: List<String>
    ): Operator<WebGPUData<*>, WebGPUData<*>>  = when (opType) {
        "Add" -> Add(name, version, attributes, inputs, outputs)
        "Constant" -> Constant(name, version, attributes, inputs, outputs)
        "ConstantOfShape" -> ConstantOfShape(name, version, attributes, inputs, outputs)
        "Div" -> Div(name, version, attributes, inputs, outputs)
        "Equal" -> Equal(name, version, attributes, inputs, outputs)
        "Flatten" -> Flatten(name, version, attributes, inputs, outputs)
        "Greater" -> Greater(name, version, attributes, inputs, outputs)
        "Less" -> Less(name, version, attributes, inputs, outputs)
        "MatMul" -> MatMul(name, version, attributes, inputs, outputs)
        "Mul" -> Mul(name, version, attributes, inputs, outputs)
        "Or" -> Or(name, version, attributes, inputs, outputs)
        "Reshape" -> Reshape(name, version, attributes, inputs, outputs)
        "Squeeze" -> Squeeze(name, version, attributes, inputs, outputs)
        "Sub" -> Sub(name, version, attributes, inputs, outputs)
        "Unsqueeze" -> Unsqueeze(name, version, attributes, inputs, outputs)
        else -> error("Unsupported operator: $opType")
    } as Operator<WebGPUData<*>, WebGPUData<*>>
}
