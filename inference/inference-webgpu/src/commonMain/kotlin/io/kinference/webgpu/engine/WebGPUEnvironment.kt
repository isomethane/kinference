package io.kinference.webgpu.engine

import io.kinference.ndarray.WebGPUState
import io.kinference.utils.webgpu.Device

expect object WebGPUEnvironment {
    suspend fun getDevice(): Device

    val gpuState: WebGPUState
}
