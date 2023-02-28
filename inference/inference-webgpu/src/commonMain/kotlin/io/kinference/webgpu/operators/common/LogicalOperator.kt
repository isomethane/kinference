package io.kinference.webgpu.operators.common

import io.kinference.attribute.Attribute
import io.kinference.data.ONNXData
import io.kinference.graph.Contexts
import io.kinference.ndarray.arrays.NDArrayInfo
import io.kinference.ndarray.arrays.WebGPUDataType
import io.kinference.operator.Operator
import io.kinference.operator.OperatorInfo
import io.kinference.webgpu.data.tensor.WebGPUTensor

abstract class LogicalOperator(
    name: String, info: OperatorInfo, attributes: Map<String, Attribute<Any>>, inputs: List<String>, outputs: List<String>
) : CachingShaderOperator(name, info, attributes, inputs, outputs) {
    abstract fun operation(input0: String, input1: String, output: String): String

    override fun <D : ONNXData<*, *>> operatorImplementation(inputInfo: List<NDArrayInfo?>, contexts: Contexts<D>): Operator<WebGPUTensor, WebGPUTensor> =
        when {
            inputInfo[0]!!.shape.contentEquals(inputInfo[1]!!.shape) -> ArithmeticOperatorWithoutBroadcast(
                this::operation, inputInfo[0]!!, WebGPUDataType.INT32,
                name = name, info = info, attributes = attributes, inputs = inputs, outputs = outputs
            )
            else -> ArithmeticOperatorWithBroadcast(
                this::operation, inputInfo[0]!!, inputInfo[1]!!, WebGPUDataType.INT32,
                name = name, info = info, attributes = attributes, inputs = inputs, outputs = outputs
            )
        }
}
