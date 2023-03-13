package io.kinference.ndarray.utils

const val DEFAULT_WORK_GROUP_SIZE_1D = 128

fun shapeToWorkSize(shape: IntArray): IntArray =
    shape.reversedArray().let {
        intArrayOf(it.getOrElse(0) { 1 }, it.getOrElse(1) { 1 }, it.drop(2).fold(1, Int::times))
    }
