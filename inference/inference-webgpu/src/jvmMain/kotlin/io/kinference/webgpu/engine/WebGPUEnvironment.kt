package io.kinference.webgpu.engine

import io.kinference.ndarray.WebGPUState
import io.kinference.utils.webgpu.Device
import io.kinference.utils.webgpu.WebGPUInstance
import kotlinx.coroutines.sync.Mutex
import kotlinx.coroutines.sync.withLock
import java.util.concurrent.atomic.AtomicReference

actual object WebGPUEnvironment {
    private val device = AtomicReference<Device>()
    private val initMutex = Mutex()

    private var state: WebGPUState? = null

    actual suspend fun getDevice(): Device {
        val currentDevice: Device? = device.get()
        if (currentDevice != null) {
            return currentDevice
        }
        initMutex.withLock {
            if (device.get() == null) {
                device.set(WebGPUInstance.requestAdapter().requestDevice())
                state = WebGPUState(device.get())
            }
        }
        return device.get()
    }

    actual val gpuState: WebGPUState
        get() = state!!
}
