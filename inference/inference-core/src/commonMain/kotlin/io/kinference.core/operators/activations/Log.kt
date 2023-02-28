package io.kinference.core.operators.activations

import io.kinference.attribute.Attribute
import io.kinference.core.KIONNXData
import io.kinference.graph.Contexts
import io.kinference.operator.*
import io.kinference.ndarray.arrays.*
import io.kinference.primitives.types.DataType
import kotlin.math.ln
import kotlin.time.ExperimentalTime

sealed class Log(name: String, info: OperatorInfo, attributes: Map<String, Attribute<Any>>, inputs: List<String>, outputs: List<String>) : Activation(name, info, attributes, inputs, outputs) {
    companion object {
        private val DEFAULT_VERSION = VersionInfo(sinceVersion = 6)

        operator fun invoke(name: String, version: Int?, attributes: Map<String, Attribute<Any>>, inputs: List<String>, outputs: List<String>) = when (version ?: DEFAULT_VERSION.sinceVersion) {
            in LogVer6.VERSION.asRange() -> LogVer6(name, attributes, inputs, outputs)
            else -> error("Unsupported version of Log operator: $version")
        }

        val activateFloat = object : FloatMap {
            override fun apply(value: Float): Float = ln(value)
        }

        val activateDouble = object : DoubleMap {
            override fun apply(value: Double): Double = ln(value)
        }
    }
}

@ExperimentalTime
class LogVer6(name: String, attributes: Map<String, Attribute<Any>> = emptyMap(), inputs: List<String>, outputs: List<String>) : Log(name, INFO, attributes, inputs, outputs) {
    companion object {
        private val TYPE_CONSTRAINTS = FLOAT_DATA_TYPES

        private val INPUT_INFO = listOf(IOInfo(0, TYPE_CONSTRAINTS, "input", optional = false))
        private val OUTPUT_INFO = listOf(IOInfo(0, TYPE_CONSTRAINTS, "output", optional = false))

        internal val VERSION = VersionInfo(sinceVersion = 6)
        private val INFO = OperatorInfo("Log", emptySet(), INPUT_INFO, OUTPUT_INFO, VERSION, OperatorInfo.DEFAULT_DOMAIN)
    }

    override fun activate(input: NDArrayCore, contexts: Contexts<KIONNXData<*>>): NDArrayCore {
        return when (val type = input.type) {
            DataType.FLOAT -> input.map(Log.activateFloat)
            DataType.DOUBLE -> input.map(Log.activateDouble)
            else -> error("Unsupported data type for this operation: $type")
        }
    }
}
