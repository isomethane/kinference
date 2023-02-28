package io.kinference.webgpu.operators.common

import io.kinference.attribute.Attribute
import io.kinference.data.ONNXData
import io.kinference.graph.Contexts
import io.kinference.ndarray.arrays.*
import io.kinference.operator.OperatorInfo
import io.kinference.utils.webgpu.*
import io.kinference.webgpu.data.tensor.WebGPUTensor
import io.kinference.webgpu.data.tensor.asTensor
import io.kinference.webgpu.engine.WebGPUEnvironment

abstract class BinaryOperator(
    name: String, info: OperatorInfo, attributes: Map<String, Attribute<Any>>, inputs: List<String>, outputs: List<String>
) : ShaderOperator(name, info, attributes, inputs, outputs) {
    override val bindGroupLayoutDescriptor: BindGroupLayoutDescriptor = BindGroupLayoutDescriptor(
        listOf(
            BindGroupLayoutEntry(0, BufferBindingLayout(BufferBindingType.ReadOnlyStorage)),
            BindGroupLayoutEntry(1, BufferBindingLayout(BufferBindingType.ReadOnlyStorage)),
            BindGroupLayoutEntry(2, BufferBindingLayout(BufferBindingType.Storage))
        )
    )

    abstract val outputShape: IntArray
    abstract val outputType: WebGPUDataType

    protected val outputInfo: NDArrayInfo
        get() = NDArrayInfo(outputShape, outputType)

    override fun <D : ONNXData<*, *>> apply(contexts: Contexts<D>, inputs: List<WebGPUTensor?>): List<WebGPUTensor?> {
        val output = NDArrayWebGPU(outputInfo, WebGPUEnvironment.gpuState)
        val bindGroup = WebGPUEnvironment.gpuState.device.createBindGroup(
            BindGroupDescriptor(
                layout = WebGPUEnvironment.gpuState.device.createBindGroupLayout(bindGroupLayoutDescriptor),
                entries = listOf(
                    BindGroupEntry(0, BufferBinding(inputs[0]!!.data.getBuffer(WebGPUEnvironment.gpuState))),
                    BindGroupEntry(1, BufferBinding(inputs[1]!!.data.getBuffer(WebGPUEnvironment.gpuState))),
                    BindGroupEntry(2, BufferBinding(output.getBuffer(WebGPUEnvironment.gpuState))),
                )
            )
        )
        performComputePass(WebGPUEnvironment.gpuState, bindGroup)
        return listOf(output.asTensor("C"))
    }
}
