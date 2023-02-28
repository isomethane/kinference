package io.kinference.webgpu.graph

import io.kinference.graph.Graph
import io.kinference.graph.GraphContext
import io.kinference.operator.OperatorSetRegistry
import io.kinference.protobuf.message.GraphProto
import io.kinference.protobuf.message.TensorProto
import io.kinference.webgpu.data.tensor.WebGPUTensor
import io.kinference.webgpu.engine.WebGPUData
import io.kinference.webgpu.operators.WebGPUOperatorFactory

class WebGPUGraph(proto: GraphProto, opSetRegistry: OperatorSetRegistry) : Graph<WebGPUData<*>>(proto, opSetRegistry, WebGPUOperatorFactory) {
    override fun makeContext(root: GraphContext<WebGPUData<*>>?): GraphContext<WebGPUData<*>> = GraphContext(root)

    override fun prepareInput(proto: TensorProto): WebGPUData<*> = WebGPUTensor.create(proto)
}
