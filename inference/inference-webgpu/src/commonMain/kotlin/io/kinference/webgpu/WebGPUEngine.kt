package io.kinference.webgpu

import io.kinference.BackendInfo
import io.kinference.InferenceEngine
import io.kinference.data.ONNXData
import io.kinference.data.ONNXDataType
import io.kinference.model.Model
import io.kinference.protobuf.*
import io.kinference.protobuf.message.ModelProto
import io.kinference.utils.CommonDataLoader
import io.kinference.webgpu.data.tensor.WebGPUTensor
import io.kinference.webgpu.model.WebGPUModel
import okio.Buffer
import okio.Path
import okio.Path.Companion.toPath

typealias WebGPUData<T> = ONNXData<T, WebGPUBackend>

object WebGPUBackend : BackendInfo(name = "WebGPU")

object WebGPUEngine : InferenceEngine<WebGPUData<*>> {
    override val info = WebGPUBackend

    private val WEBGPU_READER_CONFIG = ProtobufReader.ReaderConfig(tensorDecoder = FlatTensorDecoder)
    private fun protoReader(bytes: ByteArray) = ProtobufReader(Buffer().write(bytes), WEBGPU_READER_CONFIG)

    override fun loadData(bytes: ByteArray, type: ONNXDataType): WebGPUData<*> = when (type) {
        ONNXDataType.ONNX_TENSOR -> WebGPUTensor.create(protoReader(bytes).readTensor())
        else -> TODO()
    }

    override suspend fun loadData(path: Path, type: ONNXDataType): WebGPUData<*> {
        return loadData(CommonDataLoader.bytes(path), type)
    }

    override suspend fun loadData(path: String, type: ONNXDataType): WebGPUData<*> {
        return loadData(path.toPath(), type)
    }

    override suspend fun loadModel(bytes: ByteArray): Model<WebGPUData<*>> {
        val modelScheme = ModelProto.decode(protoReader(bytes))
        return WebGPUModel(modelScheme)
    }

    override suspend fun loadModel(path: Path): Model<WebGPUData<*>> {
        return loadModel(CommonDataLoader.bytes(path))
    }

    override suspend fun loadModel(path: String): Model<WebGPUData<*>> {
        return loadModel(path.toPath())
    }
}
