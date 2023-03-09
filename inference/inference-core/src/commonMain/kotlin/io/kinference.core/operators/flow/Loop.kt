package io.kinference.core.operators.flow

import io.kinference.core.KIONNXData
import io.kinference.attribute.Attribute
import io.kinference.core.data.tensor.*
import io.kinference.core.graph.KIGraph
import io.kinference.data.ONNXData
import io.kinference.graph.Contexts
import io.kinference.ndarray.arrays.BooleanNDArray
import io.kinference.ndarray.arrays.LongNDArray
import io.kinference.operator.*
import io.kinference.protobuf.message.AttributeProto
import io.kinference.protobuf.message.TensorProto
import kotlin.time.ExperimentalTime

sealed class Loop(name: String, info: OperatorInfo, attributes: Map<String, Attribute<Any>>, inputs: List<String>, outputs: List<String>) : Operator<KITensor, KITensor>(name, info, attributes, inputs, outputs) {
    companion object {
        private val DEFAULT_VERSION = VersionInfo(sinceVersion = 1, untilVersion = 13)

        operator fun invoke(name: String, version: Int?, attributes: Map<String, Attribute<Any>>, inputs: List<String>, outputs: List<String>) = when (version ?: DEFAULT_VERSION.sinceVersion) {
            in LoopVer1.VERSION.asRange() -> LoopVer1(name, attributes, inputs, outputs)
            else -> error("Unsupported version of Loop operator: $version")
        }
    }
}

@ExperimentalTime
class LoopVer1(name: String, attributes: Map<String, Attribute<Any>>, inputs: List<String>, outputs: List<String>) : Loop(name, INFO, attributes, inputs, outputs) {
    companion object {
        private val TYPE_CONSTRAINTS = ALL_DATA_TYPES

        private val ATTRIBUTES_INFO = listOf(
            AttributeInfo("body", setOf(AttributeProto.AttributeType.GRAPH), required = true)
        )

        private val INPUTS_INFO = listOf(
            IOInfo(0, setOf(TensorProto.DataType.INT64), "M", optional = true, scalar = true),
            IOInfo(1, setOf(TensorProto.DataType.BOOL), "cond", optional = true, scalar = true),
            VariadicIOInfo(2, TYPE_CONSTRAINTS, "v_initial")
        )

        private val OUTPUTS_INFO = listOf(VariadicIOInfo(0, TYPE_CONSTRAINTS, "v_final_and_scan_outputs", minimumArity = 1))

        internal val VERSION = VersionInfo(sinceVersion = 1, untilVersion = 13)
        private val INFO = OperatorInfo("Loop", ATTRIBUTES_INFO, INPUTS_INFO, OUTPUTS_INFO, VERSION, OperatorInfo.DEFAULT_DOMAIN)
    }

    private val body: KIGraph by attribute()

    private suspend fun inner(contexts: Contexts<KIONNXData<*>>, body: KIGraph, counter: Long, condition: Boolean, modified: MutableList<KITensor>, scans: List<MutableList<KITensor>>): Boolean {
        val inputs = ArrayList<KIONNXData<*>>().apply {
            add(LongNDArray.scalar(counter).asTensor(body.inputs[0].name))
            add(BooleanNDArray.scalar(condition).asTensor(body.inputs[1].name))
            body.inputs.drop(2).zip(modified) { input, value ->
                add(value.rename(input.name))
            }
        }

        val outputs = body.execute(inputs, contexts)
        val iterationOutputs = outputs.drop(body.inputs.size - 1)

        modified.clear()
        // remove keepgoing flag (first) and take only modified (without scans)
        for (output in outputs.drop(1).take(body.inputs.size - 2)) {
            modified.add(output as KITensor)
        }

        require(iterationOutputs.size == scans.size) { "Loop subgraph didn't provide expected output count" }
        scans.zip(iterationOutputs) { buffer, output ->
            buffer.add(output as KITensor)
        }

        return (outputs[0] as KITensor).data.singleValue() as Boolean
    }

    override suspend fun <D : ONNXData<*, *>> apply(contexts: Contexts<D>, inputs: List<KITensor?>): List<KITensor?> {
        val maxTripCount = inputs[0]?.data?.singleValue() as Long?
        val keepgoing = inputs[1]?.data?.singleValue() as Boolean?

        require(body.inputs.size == inputs.size) { "Not enough inputs for Loop subgraph\nPresent: ${inputs.size}, Expected: ${body.inputs.size}" }

        // NOTE: works as ONNX Runtime (counter and condition are ignored and not returned to results of Loop)
        val modifiedCount = body.inputs.size - 2
        val modified = inputs.drop(2).requireNoNulls().toMutableList()

        val scansCount = body.outputs.size - 1 - modifiedCount
        val scans = (0 until scansCount).map { ArrayList<KITensor>() }

        var counter = 0L
        var condition = keepgoing ?: true

        contexts as Contexts<KIONNXData<*>>
        when {
            maxTripCount == null && keepgoing == null -> {
                while (true) {
                    condition = inner(contexts, body, counter, condition, modified, scans)
                    counter += 1
                }
            }
            maxTripCount == null && keepgoing != null -> {
                while (condition) {
                    condition = inner(contexts, body, counter, condition, modified, scans)
                    counter += 1
                }
            }
            maxTripCount != null && keepgoing == null -> {
                for (counter in 0 until maxTripCount) {
                    condition = inner(contexts, body, counter, condition, modified, scans)
                }
            }
            maxTripCount != null && keepgoing != null -> {
                for (counter in 0 until maxTripCount) {
                    if (!condition) break
                    condition = inner(contexts, body, counter, condition, modified, scans)
                }
            }
        }

        return modified + scans.map { it.stack(0) }
    }
}
