package io.kinference.ndarray.environment

import io.kinference.ndarray.arrays.NDArrayInfo
import io.kinference.ndarray.arrays.TypedNDArrayData
import io.kinference.utils.webgpu.*
import kotlinx.coroutines.sync.Mutex
import kotlinx.coroutines.sync.withLock

object WebGPU {
    private var deviceRef: Device? = null
    private var gpuStateRef: WebGPUState? = null
    private var isInit: Boolean = false
    private val initMutex = Mutex()

    suspend fun init() {
        initMutex.withLock {
            if (!isInit) {
                deviceRef = WebGPUInstance.requestAdapter().requestDevice()
                gpuStateRef = WebGPUState(deviceRef!!)
                isInit = true
            }
        }
    }

    val device: Device
        get() = deviceRef ?: error("WebGPU not initialized")

    private val gpuState: WebGPUState
        get() = gpuStateRef ?: error("WebGPU not initialized")

    fun beginComputePass(descriptor: ComputePassDescriptor = ComputePassDescriptor()): ComputePassEncoder = gpuState.beginComputePass(descriptor)

    fun copyBufferToBuffer(source: Buffer, sourceOffset: Int, destination: Buffer, destinationOffset: Int, size: Int) = gpuState.copyBufferToBuffer(source, sourceOffset, destination, destinationOffset, size)

    fun enqueueCommands() = gpuState.enqueueCommands()

    fun createInitializedBuffer(info: NDArrayInfo, bufferData: TypedNDArrayData): Buffer = gpuState.createInitializedBuffer(info, bufferData)

    fun createUninitializedBuffer(info: NDArrayInfo): Buffer = gpuState.createUninitializedBuffer(info)

    fun createReadableBuffer(info: NDArrayInfo, sourceBuffer: Buffer): Buffer = gpuState.createReadableBuffer(info, sourceBuffer)
}
