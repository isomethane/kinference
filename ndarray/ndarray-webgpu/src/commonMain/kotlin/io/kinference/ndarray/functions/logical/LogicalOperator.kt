package io.kinference.ndarray.functions.logical

import io.kinference.ndarray.arrays.NDArrayInfo
import io.kinference.ndarray.arrays.WebGPUDataType
import io.kinference.ndarray.functions.CachingShaderFunction
import io.kinference.ndarray.functions.NDArrayFunction
import io.kinference.ndarray.functions.math.ArithmeticFunctionWithBroadcast
import io.kinference.ndarray.functions.math.ArithmeticFunctionWithoutBroadcast

abstract class LogicalOperator : CachingShaderFunction() {
    abstract fun operation(input0: String, input1: String, output: String): String

    override fun implementation(inputInfo: List<NDArrayInfo?>): NDArrayFunction =
        when {
            inputInfo[0]!!.shape.contentEquals(inputInfo[1]!!.shape) -> ArithmeticFunctionWithoutBroadcast(
                this::operation, inputInfo[0]!!, WebGPUDataType.INT32
            )
            else -> ArithmeticFunctionWithBroadcast(
                this::operation, inputInfo[0]!!, inputInfo[1]!!, WebGPUDataType.INT32
            )
        }
}
