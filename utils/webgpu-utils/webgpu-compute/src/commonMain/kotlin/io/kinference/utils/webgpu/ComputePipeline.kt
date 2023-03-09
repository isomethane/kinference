package io.kinference.utils.webgpu

expect class ComputePipeline {
    fun getBindGroupLayout(index: Int): BindGroupLayout
}
