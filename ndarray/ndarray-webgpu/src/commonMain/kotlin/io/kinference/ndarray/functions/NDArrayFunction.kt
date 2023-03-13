package io.kinference.ndarray.functions

import io.kinference.ndarray.arrays.NDArrayWebGPU

interface NDArrayFunction {
    suspend fun apply(inputs: List<NDArrayWebGPU?>): NDArrayWebGPU

    suspend fun apply(vararg inputs: NDArrayWebGPU?): NDArrayWebGPU = apply(inputs.toList())
}
