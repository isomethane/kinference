package io.kinference.webgpu.engine

import io.kinference.ndarray.WebGPUState
import io.kinference.utils.webgpu.Device
import io.kinference.utils.webgpu.WebGPUInstance

actual object WebGPUEnvironment {
    private var device: Device? = null
    private var state: WebGPUState? = null

    actual suspend fun getDevice(): Device {
        if (device == null) {
            device = WebGPUInstance.requestAdapter().requestDevice()
            state = WebGPUState(device!!)
        }
        return device!!
    }

    actual val gpuState: WebGPUState
        get() = state!!
}
