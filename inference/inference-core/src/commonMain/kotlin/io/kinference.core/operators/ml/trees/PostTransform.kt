package io.kinference.core.operators.ml.trees

import io.kinference.ndarray.arrays.FloatNDArray
import io.kinference.ndarray.arrays.MutableFloatNDArray
import kotlin.coroutines.EmptyCoroutineContext
import kotlin.time.ExperimentalTime

//TODO: SOFTMAX_ZERO, LOGISTIC, PROBIT
@ExperimentalTime
sealed class PostTransform {
    abstract fun apply(array: MutableFloatNDArray): FloatNDArray

    object None : PostTransform() {
        override fun apply(array: MutableFloatNDArray) = array
    }

    object SoftmaxTransform : PostTransform() {
        override fun apply(array: MutableFloatNDArray): FloatNDArray {
            // TODO: coroutines in softmax can't be used without context here
            return array.softmax(axis = -1, EmptyCoroutineContext)
        }
    }

    companion object {
        operator fun get(name: String) = when (name) {
            "NONE" -> None
            "SOFTMAX" -> SoftmaxTransform
            else -> error("Unsupported post-transformation: $name")
        }
    }
}
