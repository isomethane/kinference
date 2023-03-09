package io.kinference.webgpu.engine

import io.kinference.ndarray.WebGPUState
import io.kinference.utils.webgpu.Device
import io.kinference.utils.webgpu.WebGPUInstance

actual object WebGPU {
    private var isInit: Boolean = false
    private var deviceRef: Device? = null
    private var gpuStateRef: WebGPUState? = null

    actual suspend fun init() {
        if (!isInit) {
            deviceRef = WebGPUInstance.requestAdapter().requestDevice()
            gpuStateRef = WebGPUState(deviceRef!!)
            isInit = true
        }
    }

    actual val device: Device
        get() {
            checkInit()
            return deviceRef!!
        }

    actual val gpuState: WebGPUState
        get() {
            checkInit()
            return gpuStateRef!!
        }

    private fun checkInit() {
        if (!isInit) {
            error("WebGPU not initialized")
        }
    }
}
