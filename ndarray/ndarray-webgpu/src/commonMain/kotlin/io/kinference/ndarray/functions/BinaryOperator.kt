package io.kinference.ndarray.functions

import io.kinference.ndarray.arrays.*
import io.kinference.ndarray.environment.WebGPU
import io.kinference.utils.webgpu.*

abstract class BinaryOperator : ShaderFunction() {
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

    override suspend fun apply(inputs: List<NDArrayWebGPU?>): NDArrayWebGPU {
        val output = NDArrayWebGPU(outputInfo)
        val bindGroup = WebGPU.device.createBindGroup(
            BindGroupDescriptor(
                layout = WebGPU.device.createBindGroupLayout(bindGroupLayoutDescriptor),
                entries = listOf(
                    BindGroupEntry(0, BufferBinding(inputs[0]!!.getBuffer())),
                    BindGroupEntry(1, BufferBinding(inputs[1]!!.getBuffer())),
                    BindGroupEntry(2, BufferBinding(output.getBuffer())),
                )
            )
        )
        performComputePass(bindGroup)
        return output
    }
}
