package io.kinference.webgpu.engine

import io.kinference.ndarray.WebGPUState
import io.kinference.utils.webgpu.*

expect object WebGPU {
    suspend fun init()

    val device: Device

    val gpuState: WebGPUState
}

fun WebGPU.beginComputePass(descriptor: ComputePassDescriptor = ComputePassDescriptor()): ComputePassEncoder = gpuState.beginComputePass(descriptor)
