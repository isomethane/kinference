package io.kinference.core.model

import io.kinference.core.graph.*
import io.kinference.data.ONNXDataAdapter
import io.kinference.model.Model
import io.kinference.profiler.*
import io.kinference.protobuf.message.ModelProto
import kotlin.time.ExperimentalTime

@ExperimentalTime
class KIModel<T> internal constructor(proto: ModelProto, private val adapter: ONNXDataAdapter<T>) : Model<T>, Profilable {
    val graph = Graph(proto.graph!!)
    val name: String = "${proto.domain}:${proto.modelVersion}"

    private val profiles: MutableList<ProfilingContext> = ArrayList()
    override fun addContext(name: String): ProfilingContext = ProfilingContext("Model $name").apply { profiles.add(this) }
    override fun analyzeProfilingResults(): ProfileAnalysisEntry = profiles.analyze("Model $name")
    override fun resetProfiles() = profiles.clear()

    override fun predict(input: Map<String, T>, profile: Boolean): Map<String, T> {
        val context = if (profile) addContext("Model $name") else null
        val onnxInput = input.map { (name, data) -> adapter.toONNXData(name, data) }
        val execResult = graph.execute(onnxInput, profilingContext = context)
        return execResult.associate { it.name!! to adapter.fromONNXData(it) }
    }
}