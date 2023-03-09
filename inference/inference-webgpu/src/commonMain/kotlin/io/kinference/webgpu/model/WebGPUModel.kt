package io.kinference.webgpu.model

import io.kinference.model.Model
import io.kinference.operator.OperatorSetRegistry
import io.kinference.protobuf.message.ModelProto
import io.kinference.utils.LoggerFactory
import io.kinference.webgpu.data.tensor.WebGPUTensor
import io.kinference.webgpu.engine.WebGPUData
import io.kinference.webgpu.engine.WebGPUEnvironment
import io.kinference.webgpu.graph.WebGPUGraph

class WebGPUModel(val name: String, val opSet: OperatorSetRegistry, val graph: WebGPUGraph) : Model<WebGPUData<*>> {
    override suspend fun predict(input: List<WebGPUData<*>>, profile: Boolean): Map<String, WebGPUData<*>> {
        if (profile) logger.warning { "Profiling of models running on WebGPU backend is not supported" }

        WebGPUEnvironment.getDevice()
        val outputs = graph.execute(input).associateBy { it.name.orEmpty() }
        outputs.forEach { (_, value) ->
            if (value is WebGPUTensor) {
                value.data.requestData(WebGPUEnvironment.gpuState)
            }
        }
        outputs.forEach { (_, value) ->
            if (value is WebGPUTensor) {
                value.data.finalizeOutputNDArray(WebGPUEnvironment.gpuState)
            }
        }
        return outputs
    }

    override fun close() {
        graph.close()
    }

    companion object {
        suspend operator fun invoke(proto: ModelProto): WebGPUModel {
            val name = "${proto.domain}:${proto.modelVersion}"
            val opSet = OperatorSetRegistry(proto.opSetImport)
            val graph = WebGPUGraph(proto.graph!!, opSet)
            return WebGPUModel(name, opSet, graph)
        }

        private val logger = LoggerFactory.create("io.kinference.webgpu.model.WebGPUModel")
    }
}
