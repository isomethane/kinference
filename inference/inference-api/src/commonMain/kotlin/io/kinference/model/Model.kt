package io.kinference.model

import io.kinference.InferenceEngine
import io.kinference.OptimizableEngine
import io.kinference.data.ONNXData

/**
 * Interface for KInference executable models.
 *
 * @param T type of model data representation.
 */
interface Model<T : ONNXData<*, *>> {
    /**
     * Executes the model with profiling and given context.
     * If [profile] flag is true, prints profiling info after the model pass.
     */
    fun predict(input: List<T>, profile: Boolean = false, executionContext: ExecutionContext? = null): Map<String, T>
    suspend fun predictSuspend(input: List<T>, profile: Boolean = false, executionContext: ExecutionContext? = null): Map<String, T> =
        predict(input, profile, executionContext)

    companion object {
        /**
         * Reads model using given inference engine.
         * Model should be previously loaded as a [ByteArray].
         */
        fun <T : ONNXData<*, *>> load(bytes: ByteArray, engine: InferenceEngine<T>) = engine.loadModel(bytes)

        /**
         * Reads model using given inference engine.
         * Model should be previously loaded as a [ByteArray].
         * If [optimize] flag is true, runs available optimizations on the given model.
         */
        fun <T : ONNXData<*, *>> load(bytes: ByteArray, engine: OptimizableEngine<T>, optimize: Boolean) = engine.loadModel(bytes, optimize)
    }
}
