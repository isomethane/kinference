package io.kinference.utils.webgpu

import io.kinference.utils.wgpu.jnr.WGPUComputePipeline

actual class ComputePipeline(val wgpuComputePipeline: WGPUComputePipeline) {
    actual fun getBindGroupLayout(index: Int): BindGroupLayout = BindGroupLayout(
        WebGPUInstance.wgpuNative.wgpuComputePipelineGetBindGroupLayout(wgpuComputePipeline, index.toLong())
    )
}
