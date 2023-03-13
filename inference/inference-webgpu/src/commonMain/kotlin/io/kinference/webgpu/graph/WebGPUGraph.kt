package io.kinference.webgpu.graph

import io.kinference.graph.*
import io.kinference.operator.Operator
import io.kinference.operator.OperatorSetRegistry
import io.kinference.protobuf.message.GraphProto
import io.kinference.protobuf.message.TensorProto
import io.kinference.webgpu.WebGPUData
import io.kinference.webgpu.data.tensor.WebGPUTensor
import io.kinference.webgpu.operators.WebGPUOperatorFactory

class WebGPUGraph(
    proto: GraphProto,
    operators: ArrayList<Operator<WebGPUData<*>, WebGPUData<*>>>,
    valueOrderInfo: GraphValueOrderInfo,
) : Graph<WebGPUData<*>>(proto, operators, valueOrderInfo) {
    override fun makeContext(root: GraphContext<WebGPUData<*>>?): GraphContext<WebGPUData<*>> = GraphContext(root)

    override fun prepareInput(proto: TensorProto): WebGPUData<*> = WebGPUTensor.create(proto)

    companion object {
        suspend operator fun invoke(proto: GraphProto, opSetRegistry: OperatorSetRegistry): WebGPUGraph {
            val valueOrderInfo = GraphValueOrderInfo()
            val nodes = proto.collectOperators<WebGPUData<*>>(valueOrderInfo)
            val operators = ArrayList<Operator<WebGPUData<*>, WebGPUData<*>>>(nodes.size).apply {
                for (node in nodes) {
                    add(WebGPUOperatorFactory.create(node.proto, opSetRegistry))
                }
            }

            return WebGPUGraph(proto, operators, valueOrderInfo)
        }
    }
}
