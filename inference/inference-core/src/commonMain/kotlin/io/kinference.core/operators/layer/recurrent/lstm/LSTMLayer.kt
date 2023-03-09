package io.kinference.core.operators.layer.recurrent.lstm

import io.kinference.ndarray.arrays.*
import io.kinference.core.operators.activations.Activation
import io.kinference.ndarray.extensions.*
import io.kinference.primitives.types.DataType

import kotlin.time.ExperimentalTime

@OptIn(ExperimentalTime::class)
class LSTMLayer(hiddenSize: Int, activations: List<String>, direction: String) : LSTMLayerBase(hiddenSize, activations, direction) {
    init {
        require(activations.size == 3)
    }

    override suspend fun apply(
        input: AbstractLSTMInput,
        weights: AbstractLSTMWeights,
        recurrentWeights: AbstractLSTMWeights,
        bias: NumberNDArrayCore?,
        sequenceLens: IntNDArray?,
        initialHiddenState: NumberNDArrayCore?,
        initialCellState: NumberNDArrayCore?,
        peepholes: NumberNDArrayCore?,
        dataType: DataType
    ): LSTMLayerOutput {
        val h = Activation.create(activations[2], dataType)

        val seqLength = input.data.shape[0]
        val batchSize = input.data.shape[1]
        val outputArray = allocateNDArray(dataType, intArrayOf(seqLength, 1, batchSize, hiddenSize)) as MutableNumberNDArrayCore

        val initHiddenState = (initialHiddenState?.toMutable() ?: allocateNDArray(dataType, intArrayOf(1, batchSize, hiddenSize))) as MutableNumberNDArrayCore
        val initHiddenStateAsLSTMInput = arrayOf(input.recreate(initHiddenState.view(0)))

        val lstmStates = LSTMStates(
            LSTMCellState(initialCellState, dataType, 1, batchSize, hiddenSize),
            LSTMHiddenState(initHiddenState, initHiddenStateAsLSTMInput, listOf(h))
        )

        val lstmGates = LSTMGates.create(
            weights.view(0),
            recurrentWeights.view(0),
            bias?.view(0),
            peepholes?.view(0),
            batchSize, hiddenSize, dataType
        )

        apply(input, outputArray, lstmStates, lstmGates, sequenceLens, 0, seqLength, batchSize, dataType)

        return LSTMLayerOutput(outputArray, lstmStates.hiddenState.data, lstmStates.cellState.data)
    }

    suspend fun apply(
        input: AbstractLSTMInput,
        output: MutableNumberNDArrayCore,
        lstmStates: LSTMStates,
        lstmGates: LSTMGates,
        sequenceLens: IntNDArray?,
        numDirection: Int,
        seqLength: Int,
        batchSize: Int,
        dataType: DataType
    ) {
        val (f, g) = activations.map { Activation.create(it, dataType) }

        val seqLens = sequenceLens?.array?.toArray() ?: IntArray(batchSize) { seqLength }
        val seqRange = if (direction == "forward") 0 until seqLength else (0 until seqLength).reversed()

        suspend fun wrapper(seqNum: Int, body: suspend (inner: suspend () -> Unit) -> Unit = { it() }) {
            for (batchNum in 0 until batchSize) {
                if (seqNum >= seqLens[batchNum]) continue
                body {
                    val localInput = input.view(seqNum, batchNum)
                    lstmGates.input.compute(localInput, lstmStates, f, numDirection, batchNum)
                    lstmGates.forget.compute(localInput, lstmStates, f, numDirection, batchNum)
                    lstmGates.cell.compute(localInput, lstmStates, g, numDirection, batchNum)
                    lstmStates.cellState.compute(lstmGates, numDirection, batchNum)
                    lstmGates.output.compute(localInput, lstmStates, f, numDirection, batchNum)
                    lstmStates.hiddenState.compute(lstmGates, lstmStates.cellState, numDirection, batchNum)
                    val outputVector = lstmStates.hiddenState.getVectorRaw(numDirection, batchNum)

                    output.viewMutable(seqNum, numDirection, batchNum).copyFrom(0, outputVector)
                }
            }

            lstmStates.hiddenState.update(numDirection)
        }

        //TODO: research optimal batchSize for run with coroutines
        for (seqNum in seqRange) {
//            if (batchSize > 1) {
//                runBlocking(executionContext.asCoroutineContext()) { wrapper(seqNum) { launch { it() } }  }
//            } else {
                wrapper(seqNum)
//            }
        }
    }
}
