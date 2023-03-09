package io.kinference.webgpu.utils

import io.kinference.ndarray.arrays.*
import io.kinference.ndarray.extensions.*
import io.kinference.utils.webgpu.Buffer
import io.kinference.webgpu.engine.WebGPU

operator fun NDArrayWebGPU.Companion.invoke(info: NDArrayInfo): NDArrayWebGPU = NDArrayWebGPU(info, WebGPU.gpuState)

fun NDArrayWebGPU.getBuffer(): Buffer = getBuffer(WebGPU.gpuState)

suspend fun NDArrayWebGPU.finalizeOutputNDArray() = finalizeOutputNDArray(WebGPU.gpuState)

fun NDArrayWebGPU.requestData(): Unit = requestData(WebGPU.gpuState)

fun NDArrayWebGPU.reshape(newShape: IntArray): NDArrayWebGPU = reshape(newShape, WebGPU.gpuState)

suspend fun NDArrayWebGPU.reshape(tensorShape: NDArrayWebGPU): NDArrayWebGPU = reshape(tensorShape, WebGPU.gpuState)

fun NDArrayWebGPU.squeeze(axes: IntArray): NDArrayWebGPU = squeeze(axes, WebGPU.gpuState)

fun NDArrayWebGPU.unsqueeze(axes: IntArray): NDArrayWebGPU = unsqueeze(axes, WebGPU.gpuState)

suspend fun NDArrayWebGPU.getData(): TypedNDArrayData = getData(WebGPU.gpuState)
