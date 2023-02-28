package io.kinference.core.operators.layer.recurrent.gru

import io.kinference.graph.Contexts
import io.kinference.ndarray.arrays.*
import io.kinference.ndarray.extensions.allocateNDArray
import io.kinference.primitives.types.DataType

class BiGRULayer(hiddenSize: Int, activations: List<String>): GRULayerBase(hiddenSize, activations, "bidirectional") {
    init {
        require(activations.size == 4)
    }

    private val forwardLayer = GRULayer(hiddenSize, activations.subList(0, 2), "forward")
    private val reverseLayer = GRULayer(hiddenSize, activations.subList(2, 4), "reverse")

    override fun apply(
        input: NumberNDArrayCore,
        weights: NumberNDArrayCore,
        recurrentWeights: NumberNDArrayCore,
        bias: NumberNDArrayCore?,
        sequenceLength: IntNDArray?,
        initialHiddenState: NumberNDArrayCore?,
        dataType: DataType,
        linearBeforeReset: Boolean,
        contexts: Contexts<*>
    ): Pair<NumberNDArrayCore, NumberNDArrayCore> {
        val seqLength = input.shape[0]
        val batchSize = input.shape[1]

        val forwardGRUGates = GRUGates.create(
            weights.view(0),
            recurrentWeights.view(0),
            bias?.view(0),
            batchSize, hiddenSize, dataType, linearBeforeReset
        )

        val reverseGRUGates = GRUGates.create(
            weights.view(1),
            recurrentWeights.view(1),
            bias?.view(1),
            batchSize, hiddenSize, dataType, linearBeforeReset
        )

        val gruHiddenState = GRUHiddenState(initialHiddenState, dataType, 2, batchSize, hiddenSize)

        val outputArray = allocateNDArray(dataType, intArrayOf(seqLength, 2, batchSize, hiddenSize)) as MutableNumberNDArrayCore

        forwardLayer.apply(input, outputArray, gruHiddenState, forwardGRUGates, sequenceLength, 0, seqLength, batchSize, dataType, contexts)
        reverseLayer.apply(input, outputArray, gruHiddenState, reverseGRUGates, sequenceLength, 1, seqLength, batchSize, dataType, contexts)

        return outputArray to gruHiddenState.data
    }
}
