package io.kinference.multik

import io.kinference.core.KIONNXData
import io.kinference.core.model.KIModel
import io.kinference.data.*
import org.jetbrains.kotlinx.multik.ndarray.data.MultiArray

class KIMultikAdapter(model: KIModel) : ONNXModelAdapter<KIONNXData<*>>(model) {
    override val adapters: Map<ONNXDataType, ONNXDataAdapter<Any, KIONNXData<*>>> = mapOf(
        ONNXDataType.ONNX_TENSOR to KIMultikTensorAdapter,
        ONNXDataType.ONNX_SEQUENCE to KIMultikSequenceAdapter,
        ONNXDataType.ONNX_MAP to KIMultikMapAdapter
    ) as Map<ONNXDataType, ONNXDataAdapter<Any, KIONNXData<*>>>

    override fun <T> T.onnxType(): ONNXDataType = when (this) {
        is MultiArray<*, *> -> ONNXDataType.ONNX_TENSOR
        is List<*> -> ONNXDataType.ONNX_SEQUENCE
        is Map<*, *> -> ONNXDataType.ONNX_MAP
        else -> error("Cannot resolve ONNX data type for ${this!!::class}")
    }
}

