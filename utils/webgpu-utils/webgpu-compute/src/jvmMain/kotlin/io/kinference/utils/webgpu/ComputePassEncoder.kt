package io.kinference.utils.webgpu

import io.kinference.utils.wgpu.jnr.WGPUComputePassEncoder

actual class ComputePassEncoder(private val wgpuComputePassEncoder: WGPUComputePassEncoder) {
    actual fun dispatchWorkgroups(x: Int, y: Int, z: Int) =
        WebGPUInstance.wgpuNative.wgpuComputePassEncoderDispatchWorkgroups(
            wgpuComputePassEncoder, x.toLong(), y.toLong(), z.toLong()
        )

    actual fun dispatchWorkgroupsIndirect(indirectBuffer: Buffer, indirectOffset: Int) =
        WebGPUInstance.wgpuNative.wgpuComputePassEncoderDispatchWorkgroupsIndirect(
            wgpuComputePassEncoder, indirectBuffer.wgpuBuffer, indirectOffset.toLong()
        )

    actual fun end() = WebGPUInstance.wgpuNative.wgpuComputePassEncoderEnd(wgpuComputePassEncoder)

    actual fun setBindGroup(index: Int, bindGroup: BindGroup, dynamicOffsets: List<BufferDynamicOffset>) =
        WebGPUInstance.wgpuNative.wgpuComputePassEncoderSetBindGroup(
            wgpuComputePassEncoder,
            groupIndex = index.toLong(),
            group = bindGroup.wgpuBindGroup,
            dynamicOffsetCount = dynamicOffsets.size.toLong(),
            dynamicOffsets = dynamicOffsets.toLongArray().createPointerTo()
        )

    actual fun setBindGroup(index: Int, bindGroup: BindGroup, dynamicOffsetsData: BufferData, dynamicOffsetsDataStart: Int, dynamicOffsetsDataLength: Int) {
        WebGPUInstance.wgpuNative.wgpuComputePassEncoderSetBindGroup(
            wgpuComputePassEncoder,
            groupIndex = index.toLong(),
            group = bindGroup.wgpuBindGroup,
            dynamicOffsetCount = dynamicOffsetsDataLength.toLong(),
            dynamicOffsets = dynamicOffsetsData.pointer.slice(dynamicOffsetsDataStart.toLong() * UInt.SIZE_BYTES, dynamicOffsetsDataLength.toLong() * UInt.SIZE_BYTES)
        )
    }

    actual fun setPipeline(pipeline: ComputePipeline) = WebGPUInstance.wgpuNative.wgpuComputePassEncoderSetPipeline(
        wgpuComputePassEncoder, pipeline.wgpuComputePipeline
    )
}
