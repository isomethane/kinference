package io.kinference.utils.webgpu

actual class ComputePipeline(val gpuComputePipeline: GPUComputePipeline) {
    actual fun getBindGroupLayout(index: Int): BindGroupLayout = gpuComputePipeline.getBindGroupLayout(index.toLong())
}

external class GPUComputePipeline {
    fun getBindGroupLayout(index: Long): GPUBindGroupLayout
}
