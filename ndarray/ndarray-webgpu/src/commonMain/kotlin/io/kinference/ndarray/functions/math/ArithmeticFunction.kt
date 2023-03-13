package io.kinference.ndarray.functions.math

import io.kinference.ndarray.arrays.*
import io.kinference.ndarray.broadcasting.unsqueezeFirst
import io.kinference.ndarray.functions.*
import io.kinference.ndarray.utils.*

abstract class ArithmeticFunction : CachingShaderFunction() {
    abstract fun operation(input0: String, input1: String, output: String): String

    override fun implementation(inputInfo: List<NDArrayInfo?>): NDArrayFunction =
        when {
            inputInfo[0]!!.shape.contentEquals(inputInfo[1]!!.shape) -> ArithmeticFunctionWithoutBroadcast(this::operation, inputInfo[0]!!)
            else -> ArithmeticFunctionWithBroadcast(this::operation, inputInfo[0]!!, inputInfo[1]!!)
        }
}

class ArithmeticFunctionWithoutBroadcast(
    private val operation: (String, String, String) -> String,
    private val inputInfo: NDArrayInfo,
    override val outputType: WebGPUDataType = inputInfo.type
) : BinaryOperator() {
    override val outputShape: IntArray = inputInfo.shape

    override val shader: String
        get() = """
struct InputMatrix {
    data: array<${inputInfo.type.wgslType}>,
}

struct OutputMatrix {
    data: array<${outputInfo.type.wgslType}>,
}

@group(0) @binding(0) var<storage, read> matrix0 : InputMatrix;
@group(0) @binding(1) var<storage, read> matrix1 : InputMatrix;
@group(0) @binding(2) var<storage, read_write> matrix2 : OutputMatrix;

@compute @workgroup_size(${workGroupSize.joinToString()})
fn main(@builtin(global_invocation_id) global_id : vec3<u32>) {
    if global_id.x >= ${inputInfo.size}u {
        return;
    }
    
    ${operation("matrix0.data[global_id.x]", "matrix1.data[global_id.x]", "matrix2.data[global_id.x]")}
}
"""
    override val workGroupSize: IntArray
        get() = intArrayOf(DEFAULT_WORK_GROUP_SIZE_1D)
    override val dispatchSize: IntArray
        get() = intArrayOf(inputInfo.size divUp workGroupSize[0])
}

class ArithmeticFunctionWithBroadcast(
    private val operation: (String, String, String) -> String,
    private val inputInfo0: NDArrayInfo,
    private val inputInfo1: NDArrayInfo,
    override val outputType: WebGPUDataType = inputInfo0.type
) : BinaryOperator() {
    override val outputShape: IntArray = broadcastShape(listOf(inputInfo0.shape, inputInfo1.shape))

    override val shader: String
        get() {
            val shapes = arrayListOf(inputInfo0.shape, inputInfo1.shape, outputInfo.shape)
            val maxRank = shapes.maxOf { it.size }
            shapes.forEachIndexed { index, shape ->
                shapes[index] = unsqueezeFirst(shape, maxRank)
            }
            val bounds = shapeToWorkSize(outputInfo.shape)
            val outputShapeReversed = outputInfo.shape.reversedArray()
            val indices = outputShapeReversed.mapIndexed { index, value ->
                if (index < 2 || outputShapeReversed.size <= 3) {
                    "    let i$index = i32(global_id[$index]);"
                } else {
                    "    let i$index = (i32(global_id[2]) / ${outputShapeReversed.take(index).drop(2).fold(1, Int::times)}) % $value;"
                }
            }.joinToString("\n")

            val dataIndices = shapes.map { shape ->
                Strides(shape).strides.mapIndexed { index, value ->
                    if (shape[index] == 1) "0" else "$value * i${maxRank - index - 1}"
                }.joinToString(separator = " + ")
            }
            val (input0, input1, output) = dataIndices.mapIndexed { index, value -> "matrix${index}.data[${value}]" }

            return """
struct InputMatrix {
    data: array<${inputInfo0.type.wgslType}>,
}

struct OutputMatrix {
    data: array<${outputInfo.type.wgslType}>,
}

@group(0) @binding(0) var<storage, read> matrix0 : InputMatrix;
@group(0) @binding(1) var<storage, read> matrix1 : InputMatrix;
@group(0) @binding(2) var<storage, read_write> matrix2 : OutputMatrix;

@compute @workgroup_size(${workGroupSize.joinToString()})
fn main(@builtin(global_invocation_id) global_id : vec3<u32>) {
    if ${bounds.withIndex().joinToString(separator = " || ") { (index, value) -> "global_id[$index] >= ${value}u" }} {
        return;
    }
$indices
    
    ${operation(input0, input1, output)}
}
"""
        }

    override val workGroupSize: IntArray
        get() {
            val reversedShape = outputInfo.shape.reversedArray()
            val result = intArrayOf(DEFAULT_WORK_GROUP_SIZE_1D, 1, 1)
            for (dim in 0..1) {
                if (reversedShape.size <= dim + 1) {
                    break
                }
                while (reversedShape[dim] * 2 <= result[dim]) {
                    result[dim] /= 2
                    result[dim + 1] *= 2
                }
            }
            return result
        }
    override val dispatchSize: IntArray
        get() = (shapeToWorkSize(outputInfo.shape) zip workGroupSize).map { (dim, workSize) -> dim divUp workSize }.toIntArray()
}
