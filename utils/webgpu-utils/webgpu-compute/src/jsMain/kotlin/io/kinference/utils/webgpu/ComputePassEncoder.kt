package io.kinference.utils.webgpu

import org.khronos.webgl.Uint32Array

actual class ComputePassEncoder(val gpuComputePassEncoder: GPUComputePassEncoder) {
    actual fun dispatchWorkgroups(x: Int, y: Int, z: Int) = gpuComputePassEncoder.dispatchWorkgroups(x, y, z)

    actual fun dispatchWorkgroupsIndirect(indirectBuffer: Buffer, indirectOffset: Int) =
        gpuComputePassEncoder.dispatchWorkgroupsIndirect(indirectBuffer.gpuBuffer, indirectOffset)

    actual fun end() = gpuComputePassEncoder.end()

    actual fun setBindGroup(index: Int, bindGroup: BindGroup, dynamicOffsets: List<BufferDynamicOffset>) =
        gpuComputePassEncoder.setBindGroup(index, bindGroup, dynamicOffsets.toTypedArray())

    actual fun setBindGroup(index: Int, bindGroup: BindGroup, dynamicOffsetsData: BufferData, dynamicOffsetsDataStart: Int, dynamicOffsetsDataLength: Int) =
        gpuComputePassEncoder.setBindGroup(index, bindGroup, Uint32Array(dynamicOffsetsData.buffer), dynamicOffsetsDataStart, dynamicOffsetsDataLength)

    actual fun setPipeline(pipeline: ComputePipeline) = gpuComputePassEncoder.setPipeline(pipeline.gpuComputePipeline)
}

external class GPUComputePassEncoder {
    fun dispatchWorkgroups(x: Int, y: Int, z: Int)
    fun dispatchWorkgroupsIndirect(indirectBuffer: GPUBuffer, indirectOffset: Int)
    fun end()
    fun setBindGroup(index: Int, bindGroup: GPUBindGroup, dynamicOffsets: Array<BufferDynamicOffset>)
    fun setBindGroup(index: Int, bindGroup: GPUBindGroup, dynamicOffsetsData: Uint32Array, dynamicOffsetsDataStart: Int, dynamicOffsetsDataLength: Int)
    fun setPipeline(pipeline: GPUComputePipeline)
}
