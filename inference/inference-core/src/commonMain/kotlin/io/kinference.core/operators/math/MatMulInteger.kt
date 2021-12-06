package io.kinference.core.operators.math

import io.kinference.attribute.Attribute
import io.kinference.core.KIONNXData
import io.kinference.core.data.tensor.KITensor
import io.kinference.core.data.tensor.asTensor
import io.kinference.core.graph.ContextPrepare
import io.kinference.core.graph.KIContext
import io.kinference.data.ONNXData
import io.kinference.graph.Context
import io.kinference.profiler.ProfilingContext
import io.kinference.ndarray.arrays.*
import io.kinference.ndarray.arrays.pointers.mapTo
import io.kinference.ndarray.arrays.tiled.IntTiledArray
import io.kinference.ndarray.extensions.matmul
import io.kinference.operator.*
import kotlin.time.ExperimentalTime
import io.kinference.protobuf.message.TensorProto

sealed class MatMulInteger(info: OperatorInfo, attributes: Map<String, Attribute<Any>>, inputs: List<String>, outputs: List<String>) : Operator<KITensor, KITensor>(info, attributes, inputs, outputs) {
    companion object {
        private val DEFAULT_VERSION = VersionInfo(sinceVersion = 10)

        operator fun invoke(version: Int?, attributes: Map<String, Attribute<Any>>, inputs: List<String>, outputs: List<String>) = when (version ?: DEFAULT_VERSION.sinceVersion) {
            in MatMulIntegerVer10.VERSION.asRange() -> MatMulIntegerVer10(attributes, inputs, outputs)
            else -> error("Unsupported version of MatMulInteger operator: $version")
        }
    }
}

@ExperimentalTime
class MatMulIntegerVer10(attributes: Map<String, Attribute<Any>>, inputs: List<String>, outputs: List<String>) : MatMulInteger(INFO, attributes, inputs, outputs) {
    companion object {
        private val IN_TYPE_CONSTRAINTS = setOf(
            TensorProto.DataType.UINT8,
            TensorProto.DataType.INT8
        )

        private val OUT_TYPE_CONSTRAINTS = setOf(TensorProto.DataType.INT32)

        private val INPUTS_INFO = listOf(
            IOInfo(0, IN_TYPE_CONSTRAINTS, "A", optional = false),
            IOInfo(1, IN_TYPE_CONSTRAINTS, "B", optional = false),
            IOInfo(2, IN_TYPE_CONSTRAINTS, "a_zero_point", optional = true),
            IOInfo(3, IN_TYPE_CONSTRAINTS, "b_zero_point", optional = true)
        )

        private val OUTPUTS_INFO = listOf(IOInfo(0, OUT_TYPE_CONSTRAINTS, "Y", optional = false))

        internal val VERSION = VersionInfo(sinceVersion = 10)
        private val INFO = OperatorInfo("MatMulInteger", emptyMap(), INPUTS_INFO, OUTPUTS_INFO, VERSION, OperatorInfo.DEFAULT_DOMAIN)

        private fun NumberNDArray.toIntNDArray(): IntNDArray {
            val result = IntNDArray(IntTiledArray(this.strides), strides)
            when (this) {
                is UByteNDArray -> {
                    this.array.pointer().mapTo(result.array.pointer(), linearSize) { it.toInt() }
                }
                is ByteNDArray -> {
                    this.array.pointer().mapTo(result.array.pointer(), linearSize) { it.toInt() }
                }
                else -> error("Unsupported data type: $type")
            }

            return result
        }
    }

    object MatMulIntegerPrepare : ContextPrepare() {
        override fun appendContext(context: KIContext, initializers: List<KITensor>, operator: Operator<KIONNXData<*>, KIONNXData<*>>) {
            val leftTensor = initTensorByDefaultName("A", operator, initializers)
            val rightTensor = initTensorByDefaultName("B", operator, initializers)
            val leftZeroPoint = initTensorByDefaultName("a_zero_point", operator, initializers)
            val rightZeroPoint = initTensorByDefaultName("b_zero_point", operator, initializers)

            appendTensor(leftTensor, leftZeroPoint, context)
            appendTensor(rightTensor, rightZeroPoint, context)
        }

        internal fun prepareTensor(tensor: KITensor, zeroPoint: KITensor?): KITensor {
            val preparedTensor = if (zeroPoint == null)
                (tensor.data as NumberNDArray).toIntNDArray()
            else
                (tensor.data as NumberNDArray).withZeroPoint(zeroPoint.data as NumberNDArray)

            return preparedTensor.asTensor("prepared_${tensor.name}")
        }

        private fun appendTensor(tensor: KITensor?, zeroPoint: KITensor?, context: KIContext) {
            if (tensor != null) {
                val preparedTensor = prepareTensor(tensor, zeroPoint)
                context.putValue(preparedTensor.name!!, preparedTensor)
            }
        }
    }

    override fun <D : ONNXData<*, *>> apply(context: Context<D>, inputs: List<KITensor?>, profilingContext: ProfilingContext?): List<KITensor?> {
        val first = inputs[0]!!
        val second = inputs[1]!!
        val firstZero = inputs.getOrNull(2)
        val secondZero = inputs.getOrNull(3)

        val firstPrepared = (context.getOrNullValue("prepared_${first.name}") ?: MatMulIntegerPrepare.prepareTensor(first, firstZero)) as KITensor
        val secondPrepared = (context.getOrNullValue("prepared_${second.name}") ?: MatMulIntegerPrepare.prepareTensor(second, secondZero)) as KITensor

        val output = (firstPrepared.data as NumberNDArray) matmul (secondPrepared.data as NumberNDArray)
        return listOf(output.asTensor("y"))
    }
}
