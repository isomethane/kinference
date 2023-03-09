package io.kinference.webgpu.operators.common

import io.kinference.attribute.Attribute
import io.kinference.data.ONNXData
import io.kinference.graph.Contexts
import io.kinference.ndarray.arrays.NDArrayInfo
import io.kinference.operator.Operator
import io.kinference.operator.OperatorInfo
import io.kinference.webgpu.data.tensor.WebGPUTensor

class CachedShaderOperatorInfo(
    val inputInfo: List<NDArrayInfo?>,
    val implementation: Operator<WebGPUTensor, WebGPUTensor>
)

abstract class CachingShaderOperator(name: String, info: OperatorInfo, attributes: Map<String, Attribute<Any>>, inputs: List<String>, outputs: List<String>)
    : Operator<WebGPUTensor, WebGPUTensor>(name, info, attributes, inputs, outputs) {
    private var cachedInfo: CachedShaderOperatorInfo? = null

    abstract fun <D : ONNXData<*, *>> operatorImplementation(inputInfo: List<NDArrayInfo?>, contexts: Contexts<D>): Operator<WebGPUTensor, WebGPUTensor>

    override suspend fun <D : ONNXData<*, *>> apply(contexts: Contexts<D>, inputs: List<WebGPUTensor?>): List<WebGPUTensor?> {
        val inputInfo = inputs.map { it?.data?.info }
        if (inputInfo != cachedInfo?.inputInfo) {
            cachedInfo = CachedShaderOperatorInfo(
                inputInfo,
                operatorImplementation(inputInfo, contexts)
            )
        }
        return cachedInfo!!.implementation.apply(contexts, inputs)
    }
}

