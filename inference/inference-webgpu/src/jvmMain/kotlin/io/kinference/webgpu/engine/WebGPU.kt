package io.kinference.webgpu.engine

import io.kinference.ndarray.WebGPUState
import io.kinference.utils.webgpu.Device
import io.kinference.utils.webgpu.WebGPUInstance
import kotlinx.coroutines.sync.Mutex
import kotlinx.coroutines.sync.withLock
import java.util.concurrent.atomic.AtomicBoolean
import java.util.concurrent.atomic.AtomicReference

actual object WebGPU {
    private val deviceRef = AtomicReference<Device>()
    private val gpuStateRef = AtomicReference<WebGPUState>()
    private val initMutex = Mutex()
    private val isInit = AtomicBoolean()

    actual suspend fun init() {
        initMutex.withLock {
            if (!isInit.get()) {
                deviceRef.set(WebGPUInstance.requestAdapter().requestDevice())
                gpuStateRef.set(WebGPUState(deviceRef.get()))
                isInit.set(true)
            }
        }
    }

    actual val device: Device
        get() {
            checkInit()
            return deviceRef.get()
        }

    actual val gpuState: WebGPUState
        get() {
            checkInit()
            return gpuStateRef.get()
        }

    private fun checkInit() {
        if (!isInit.get()) {
            error("WebGPU not initialized")
        }
    }
}
