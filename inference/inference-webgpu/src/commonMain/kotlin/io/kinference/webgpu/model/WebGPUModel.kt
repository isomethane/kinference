package io.kinference.webgpu.model

import io.kinference.model.ExecutionContext
import io.kinference.model.Model
import io.kinference.operator.OperatorSetRegistry
import io.kinference.protobuf.message.ModelProto
import io.kinference.utils.LoggerFactory
import io.kinference.webgpu.data.tensor.WebGPUTensor
import io.kinference.webgpu.engine.WebGPUData
import io.kinference.webgpu.engine.WebGPUEnvironment
import io.kinference.webgpu.graph.WebGPUGraph

class WebGPUModel(proto: ModelProto) : Model<WebGPUData<*>> {
    val name: String = "${proto.domain}:${proto.modelVersion}"
    private val opSet = OperatorSetRegistry(proto.opSetImport)
    private val graph = WebGPUGraph(proto.graph!!, opSet)

    override fun predict(input: List<WebGPUData<*>>, profile: Boolean, executionContext: ExecutionContext?): Map<String, WebGPUData<*>> {
        error("Use predictSuspend with WebGPU backend")
    }

    override suspend fun predictSuspend(input: List<WebGPUData<*>>, profile: Boolean, executionContext: ExecutionContext?): Map<String, WebGPUData<*>> {
        if (profile) logger.warning { "Profiling of models running on WebGPU backend is not supported" }
        if (executionContext != null) logger.warning { "ExecutionContext for models running on WebGPU backend is not supported" }

        WebGPUEnvironment.getDevice()
        val outputs = graph.executeSuspend(input).associateBy { it.name.orEmpty() }
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

    companion object {
        private val logger = LoggerFactory.create("io.kinference.webgpu.model.WebGPUModel")
    }
}
