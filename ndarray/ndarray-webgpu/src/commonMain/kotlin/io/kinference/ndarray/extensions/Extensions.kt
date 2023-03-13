package io.kinference.ndarray.extensions

import io.kinference.ndarray.arrays.*

fun NDArrayWebGPU.indexAxis(axis: Int): Int {
    return if (axis < 0) info.rank + axis else axis
}

suspend fun NDArrayWebGPU.reshape(tensorShape: NDArrayWebGPU): NDArrayWebGPU {
    require(tensorShape.info.type == WebGPUDataType.INT32) { "Tensor shape must have INT32 type" }

    val newShape = (tensorShape.getData() as IntNDArrayData).data
    require(newShape.count { it == -1 } <= 1) { "At most one dimension of the new shape can be -1" }

    for ((i, axisShape) in newShape.withIndex()) {
        if (axisShape == 0) newShape[i] = info.shape[i]
    }

    val negativeIdx = newShape.indexOf(-1)
    if (negativeIdx != -1) {
        val elementsCount = newShape.filter { it != -1 }.fold(1, Int::times)
        newShape[negativeIdx] = info.strides.linearSize / elementsCount
    }

    return reshape(newShape)
}

fun NDArrayWebGPU.squeeze(axes: IntArray): NDArrayWebGPU {
    val actualAxes = if (axes.isNotEmpty()) {
        axes.map { indexAxis(it) }
    } else {
        info.shape.withIndex().filter { it.value == 1 }.map { it.index }
    }
    require(actualAxes.all { info.shape[it] == 1 })

    val shapeIndices = info.shape.indices - actualAxes
    val newShape = info.shape.sliceArray(shapeIndices)

    return reshape(newShape)
}

private fun indexAxisForUnsqueeze(axis: Int, shapeSize: Int): Int {
    return if (axis < 0) shapeSize + axis else axis
}

fun NDArrayWebGPU.unsqueeze(axes: IntArray): NDArrayWebGPU {
    val actualAxes = axes.map { indexAxisForUnsqueeze(it, info.rank + axes.size) }.sorted()
    val newShape = info.shape.toMutableList()
    for (axis in actualAxes) {
        newShape.add(axis, 1)
    }

    return reshape(newShape.toIntArray())
}
