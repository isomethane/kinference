package io.kinference.ndarray.functions

import io.kinference.ndarray.arrays.NDArrayInfo
import io.kinference.ndarray.arrays.NDArrayWebGPU
import kotlinx.coroutines.sync.Mutex
import kotlinx.coroutines.sync.withLock

private class ShaderFunctionInfo(
    val inputInfo: List<NDArrayInfo?>,
    val implementation: NDArrayFunction
)

abstract class CachingShaderFunction : NDArrayFunction {
    private var shaderFunctionCache: ShaderFunctionInfo? = null
    private val cacheMutex = Mutex()

    abstract fun implementation(inputInfo: List<NDArrayInfo?>): NDArrayFunction

    override suspend fun apply(inputs: List<NDArrayWebGPU?>): NDArrayWebGPU {
        return functionInfo(inputs.map { it?.info }).implementation.apply(inputs)
    }

    private suspend fun functionInfo(inputInfo: List<NDArrayInfo?>): ShaderFunctionInfo {
        cacheMutex.withLock {
            shaderFunctionCache?.let {
                if (it.inputInfo == inputInfo) {
                    return it
                }
            }
            val cache = ShaderFunctionInfo(inputInfo, implementation(inputInfo))
            return cache.also { shaderFunctionCache = cache }
        }
    }
}
