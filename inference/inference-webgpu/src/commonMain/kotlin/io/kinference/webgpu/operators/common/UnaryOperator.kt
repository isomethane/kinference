package io.kinference.webgpu.operators.common

import io.kinference.attribute.Attribute
import io.kinference.data.ONNXData
import io.kinference.graph.Contexts
import io.kinference.ndarray.arrays.*
import io.kinference.operator.OperatorInfo
import io.kinference.utils.webgpu.*
import io.kinference.webgpu.data.tensor.WebGPUTensor
import io.kinference.webgpu.data.tensor.asTensor
import io.kinference.webgpu.engine.WebGPU
import io.kinference.webgpu.utils.getBuffer
import io.kinference.webgpu.utils.invoke

abstract class UnaryOperator(
    name: String, info: OperatorInfo, attributes: Map<String, Attribute<Any>>, inputs: List<String>, outputs: List<String>
) : ShaderOperator(name, info, attributes, inputs, outputs) {
    override val bindGroupLayoutDescriptor: BindGroupLayoutDescriptor = BindGroupLayoutDescriptor(
        listOf(
            BindGroupLayoutEntry(0, BufferBindingLayout(BufferBindingType.ReadOnlyStorage)),
            BindGroupLayoutEntry(1, BufferBindingLayout(BufferBindingType.Storage))
        )
    )

    abstract val outputShape: IntArray
    abstract val outputType: WebGPUDataType

    protected val outputInfo: NDArrayInfo
        get() = NDArrayInfo(outputShape, outputType)

    override suspend fun <D : ONNXData<*, *>> apply(contexts: Contexts<D>, inputs: List<WebGPUTensor?>): List<WebGPUTensor?> {
        val output = NDArrayWebGPU(outputInfo)
        val bindGroup = WebGPU.device.createBindGroup(
            BindGroupDescriptor(
                layout = WebGPU.device.createBindGroupLayout(bindGroupLayoutDescriptor),
                entries = listOf(
                    BindGroupEntry(0, BufferBinding(inputs[0]!!.data.getBuffer())),
                    BindGroupEntry(1, BufferBinding(output.getBuffer())),
                )
            )
        )
        performComputePass(bindGroup)
        return listOf(output.asTensor("output"))
    }
}
