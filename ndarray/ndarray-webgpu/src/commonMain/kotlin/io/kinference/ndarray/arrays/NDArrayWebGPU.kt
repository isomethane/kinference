package io.kinference.ndarray.arrays

import io.kinference.ndarray.WebGPUState
import io.kinference.ndarray.unpack
import io.kinference.utils.webgpu.*

class NDArrayWebGPU private constructor(val info: NDArrayInfo, private var state: NDArrayState) {
    fun getBuffer(gpuState: WebGPUState): Buffer {
        prepareBuffer(gpuState)
        return (state as NDArrayBuffer).buffer
    }

    suspend fun finalizeOutputNDArray(gpuState: WebGPUState) {
        getData(gpuState)
        when (val currentState = state) {
            is NDArrayDestroyed -> error("Use of destroyed buffer")
            is NDArrayData -> currentState.data
            is NDArrayInitializedBuffer -> {
                state = NDArrayData(currentState.data)
            }
            is NDArrayUninitializedBuffer, is NDArrayCopyingBuffer -> error("Incorrect state")
        }
    }

    fun requestData(gpuState: WebGPUState): Unit =
        when (val currentState = state) {
            is NDArrayDestroyed -> error("Use of destroyed buffer")
            is NDArrayData, is NDArrayInitializedBuffer, is NDArrayCopyingBuffer -> {}
            is NDArrayUninitializedBuffer -> {
                state = NDArrayCopyingBuffer(
                    sourceBuffer = currentState.buffer,
                    destinationBuffer = gpuState.createReadableBuffer(info, currentState.buffer)
                )
            }
        }

    fun getReadyData(): TypedNDArrayData = (state as NDArrayData).data

    suspend fun getData(gpuState: WebGPUState): TypedNDArrayData =
        when (val currentState = state) {
            is NDArrayDestroyed -> error("Use of destroyed buffer")
            is NDArrayData -> currentState.data
            is NDArrayInitializedBuffer -> currentState.data
            is NDArrayUninitializedBuffer -> {
                requestData(gpuState)
                getData(gpuState)
            }
            is NDArrayCopyingBuffer -> {
                gpuState.enqueueCommands()
                currentState.destinationBuffer.mapAsync(MapModeFlags(MapMode.Read))
                val data = currentState.destinationBuffer.getMappedRange().unpack(info)
                state = NDArrayInitializedBuffer(data, currentState.sourceBuffer)
                data
            }
        }

    private fun prepareBuffer(gpuState: WebGPUState): Unit =
        when (val currentState = state) {
            is NDArrayDestroyed -> error("Use of destroyed buffer")
            is NDArrayData -> {
                state = NDArrayInitializedBuffer(currentState.data, gpuState.createInitializedBuffer(info, currentState.data))
            }
            is NDArrayBuffer -> {}
        }

    fun destroy() {
        val currentState = state
        if (currentState is NDArrayBuffer) {
            currentState.buffer.destroy()
        }
        state = NDArrayDestroyed
    }

    fun reshape(newShape: IntArray, gpuState: WebGPUState): NDArrayWebGPU {
        val newInfo = NDArrayInfo(newShape, info.type)
        return when (val currentState = state) {
            is NDArrayDestroyed -> error("Use of destroyed buffer")
            is NDArrayData -> NDArrayWebGPU(newInfo, NDArrayData(currentState.data))
            is NDArrayInitializedBuffer -> NDArrayWebGPU(newInfo, NDArrayData(currentState.data))
            is NDArrayBuffer -> {
                val newBuffer = gpuState.createUninitializedBuffer(newInfo)
                gpuState.copyBufferToBuffer(currentState.buffer, 0, newBuffer, 0, newInfo.size)
                NDArrayWebGPU(newInfo, NDArrayUninitializedBuffer(newBuffer))
            }
        }
    }

    companion object {
        fun intNDArray(info: NDArrayInfo, data: IntArray): NDArrayWebGPU = NDArrayWebGPU(info, IntNDArrayData(data))
        fun uintNDArray(info: NDArrayInfo, data: UIntArray): NDArrayWebGPU = NDArrayWebGPU(info, UIntNDArrayData(data))
        fun floatNDArray(info: NDArrayInfo, data: FloatArray): NDArrayWebGPU = NDArrayWebGPU(info, FloatNDArrayData(data))

        fun scalar(value: Int) = intNDArray(NDArrayInfo(intArrayOf(), WebGPUDataType.INT32), intArrayOf(value))
        fun scalar(value: UInt) = uintNDArray(NDArrayInfo(intArrayOf(), WebGPUDataType.UINT32), uintArrayOf(value))
        fun scalar(value: Float) = floatNDArray(NDArrayInfo(intArrayOf(), WebGPUDataType.FLOAT32), floatArrayOf(value))

        operator fun invoke(info: NDArrayInfo, data: TypedNDArrayData): NDArrayWebGPU = NDArrayWebGPU(info, NDArrayData(data))

        operator fun invoke(info: NDArrayInfo, gpuState: WebGPUState): NDArrayWebGPU =
            NDArrayWebGPU(info, NDArrayUninitializedBuffer(gpuState.createUninitializedBuffer(info)))
    }
}
