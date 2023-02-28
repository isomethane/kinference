package io.kinference.tfjs.operators.math

import io.kinference.attribute.Attribute
import io.kinference.data.ONNXData
import io.kinference.graph.Contexts
import io.kinference.ndarray.arrays.NumberNDArrayTFJS
import io.kinference.ndarray.extensions.matMul
import io.kinference.ndarray.extensions.tidyNDArray
import io.kinference.operator.*
import io.kinference.protobuf.message.TensorProto
import io.kinference.tfjs.data.tensors.TFJSTensor
import io.kinference.tfjs.data.tensors.asTensor

sealed class MatMulInteger(name: String, info: OperatorInfo, attributes: Map<String, Attribute<Any>>, inputs: List<String>, outputs: List<String>) :
    Operator<TFJSTensor, TFJSTensor>(name, info, attributes, inputs, outputs) {
    companion object {
        private val DEFAULT_VERSION = VersionInfo(sinceVersion = 10)

        operator fun invoke(name: String, version: Int?, attributes: Map<String, Attribute<Any>>, inputs: List<String>, outputs: List<String>) =
            when (version ?: DEFAULT_VERSION.sinceVersion) {
                in MatMulIntegerVer10.VERSION.asRange() -> MatMulIntegerVer10(name, attributes, inputs, outputs)
                else -> error("Unsupported version of MatMulInteger operator: $version")
            }
    }
}

class MatMulIntegerVer10(name: String, attributes: Map<String, Attribute<Any>>, inputs: List<String>, outputs: List<String>) :
    MatMulInteger(name, INFO, attributes, inputs, outputs) {
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
    }

    override fun <D : ONNXData<*, *>> apply(contexts: Contexts<D>, inputs: List<TFJSTensor?>): List<TFJSTensor?> {
        val left = inputs[0]!!.data as NumberNDArrayTFJS
        val right = inputs[1]!!.data as NumberNDArrayTFJS
        val leftZP = inputs.getOrNull(2)?.data as? NumberNDArrayTFJS
        val rightZP = inputs.getOrNull(3)?.data as? NumberNDArrayTFJS
        val output = tidyNDArray {
            val leftWithZp = if (leftZP != null) left - leftZP else left
            val rightWithZp = if (rightZP != null) right - rightZP else right

            val (leftExpanded, rightExpanded) = MatMul.expandTensors(leftWithZp, rightWithZp)

            return@tidyNDArray leftExpanded.matmul(rightExpanded)
        }

        return listOf(output.asTensor("Y"))
    }
}
