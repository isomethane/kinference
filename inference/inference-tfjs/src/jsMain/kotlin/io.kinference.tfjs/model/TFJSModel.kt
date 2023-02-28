package io.kinference.tfjs.model

import io.kinference.model.ExecutionContext
import io.kinference.model.Model
import io.kinference.operator.OperatorSetRegistry
import io.kinference.protobuf.message.ModelProto
import io.kinference.tfjs.TFJSData
import io.kinference.tfjs.graph.TFJSGraph
import io.kinference.utils.LoggerFactory
import kotlin.time.ExperimentalTime

@ExperimentalTime
class TFJSModel(proto: ModelProto) : Model<TFJSData<*>> {
    private val opSet = OperatorSetRegistry(proto.opSetImport)
    val graph = TFJSGraph(proto.graph!!, opSet)
    val name: String = "${proto.domain}:${proto.modelVersion}"

   /* companion object {
        fun load(bytes: ByteArray): TFJSModel {
            val modelScheme = ModelProto.decode(bytes)
            return TFJSModel(modelScheme)
        }
    }*/

    override fun predict(input: List<TFJSData<*>>, profile: Boolean, executionContext: ExecutionContext?): Map<String, TFJSData<*>> {
        if (profile) logger.warning { "Profiling of models running on TFJS backend is not supported" }
        if (executionContext != null) logger.warning { "ExecutionContext for models running on TFJS backend is not supported" }
        return graph.execute(input).associateBy { it.name.orEmpty() }
    }

    companion object {
        private val logger = LoggerFactory.create("io.kinference.tfjs.model.TFJSModel")
    }
}
