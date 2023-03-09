@file:GeneratePrimitives(DataType.NUMBER)

package io.kinference.ndarray.extensions

import io.kinference.ndarray.*
import io.kinference.ndarray.arrays.*
import io.kinference.ndarray.arrays.pointers.accept
import io.kinference.ndarray.arrays.pointers.acceptWithRecursive
import io.kinference.ndarray.arrays.tiled.*
import io.kinference.primitives.annotations.*
import io.kinference.primitives.types.*
import kotlinx.coroutines.coroutineScope
import kotlinx.coroutines.launch
import kotlin.coroutines.CoroutineContext
import kotlin.coroutines.EmptyCoroutineContext
import kotlin.math.*

fun erf(value: PrimitiveType): PrimitiveType {
    val sign = value.toDouble().sign
    val doubleValue = abs(value.toDouble())
    val t = 1 / (1 + ERF_P_VALUE * doubleValue)

    val sum = t * (ERF_COEF[0] + t * (ERF_COEF[1] + t * (ERF_COEF[2] + t * (ERF_COEF[3] + t * ERF_COEF[4]))))

    return (sign * (1.0 - sum * exp(-doubleValue * doubleValue))).toPrimitive()
}


@SpecifyPrimitives(include = [DataType.BYTE, DataType.UBYTE])
@BindPrimitives(type1 = [DataType.BYTE, DataType.UBYTE])
suspend fun PrimitiveNDArray.quantizeDot(other: @BindPrimitives.Type1 PrimitiveNDArray, destination: MutableFloatNDArray, zeroPointA: Int = 0, zeroPointB: Int = 0, scale: Float = 1f, coroutineContext: CoroutineContext = EmptyCoroutineContext): MutableFloatNDArray {
    val M = this.shape[0]

    suspend fun wrapper(body: suspend (inner: suspend () -> Unit) -> Unit = { it() }) {
        for (rdBlockNum in 0 until destination.blocksInRow) {
            body {
                for (i in 0 until M) {
                    val dBlockOffset = i * destination.blocksInRow
                    val lBlockOffset = i * this.blocksInRow

                    var k = 0
                    for (lBlockNum in 0 until this.blocksInRow) {
                        val lBlock = this.array.blocks[lBlockOffset + lBlockNum]
                        for (lInd in lBlock.indices) {
                            val temp = lBlock[lInd].toInt() - zeroPointA
                            val rBlockOffset = k * other.blocksInRow
                            val rBlock = other.array.blocks[rBlockOffset + rdBlockNum]
                            val dBlock = destination.array.blocks[dBlockOffset + rdBlockNum]
                            for (idx in rBlock.indices) {
                                dBlock[idx] += (temp * (rBlock[idx].toInt() - zeroPointB)) * scale
                            }
                            k++
                        }
                    }
                }
            }
        }
    }

    if (other.blocksInRow > 1) {
        coroutineScope { wrapper { launch { it() } } }
    } else {
        wrapper()
    }

    return destination
}


@SpecifyPrimitives(include = [DataType.BYTE, DataType.UBYTE, DataType.INT])
suspend fun PrimitiveNDArray.withZeroPoint(zeroPoint: PrimitiveNDArray): IntNDArray {
    return if (zeroPoint.linearSize == 1) {
        val zero = zeroPoint.array.blocks[0][0].toInt()
        val arr = IntTiledArray(this.strides)
        arr.pointer().accept(array.pointer(), arr.size) { _, src -> src.toInt() - zero }
        IntNDArray(arr, strides)
    } else {
        val arr = IntTiledArray(strides)
        arr.pointer().acceptWithRecursive(this.array.pointer(), zeroPoint.array.pointer(), arr.size) { _, src, zero -> src.toInt() - zero.toInt() }
        IntNDArray(arr, strides)
    }
}

@SpecifyPrimitives(include = [DataType.BYTE, DataType.UBYTE])
suspend fun PrimitiveNDArray.dequantize(zeroPoint: PrimitiveNDArray?, scale: FloatNDArray, axis: Int?): FloatNDArray {
    val zeros = zeroPoint?.array
    val output = MutableFloatNDArray(FloatTiledArray(this.array.size, this.array.blockSize), this.strides)

    when {
        canDequantizePerTensor(zeroPoint, scale) -> {
            val zero = if (zeros == null) 0f else zeros.blocks[0][0].toFloat()
            val sc = scale.array.blocks[0][0]

            output.array.pointer().accept(this.array.pointer(), output.linearSize) { _, src -> (src.toFloat() - zero) * sc }
        }
        canDequantizePerAxis(axis!!, zeroPoint, scale) -> {
            val actualAxis = indexAxis(axis)
            val blockCount = computeBlockSize(toDim = actualAxis)
            val blockSize = computeBlockSize(fromDim = actualAxis + 1)
            var outOffset = 0
            repeat(blockCount) {
                val zeroPointer = zeros?.pointer()
                val scalePointer = scale.array.pointer()
                for (i in 0 until shape[actualAxis]) {
                    val zero = zeroPointer?.getAndIncrement()?.toFloat() ?: 0f
                    val sc = scalePointer.getAndIncrement()

                    output.array.pointer(outOffset).accept(this.array.pointer(outOffset), blockSize) { _, src -> (src.toFloat() - zero) * sc }
                    outOffset += blockSize
                }
            }
        }
        else -> error("Cannot perform dequantization. Scale and zero point tensors should be either scalars or 1D tensors containing ${shape[axis]} elements")
    }

    return output
}

@SpecifyPrimitives(include = [DataType.FLOAT, DataType.DOUBLE])
suspend fun PrimitiveNDArray.dotTransposedWithAlpha(alpha: Double, other: NumberNDArray, destination: MutableNumberNDArray): MutableNumberNDArray {
    other as PrimitiveNDArray; destination as MutablePrimitiveNDArray

    @Suppress("NAME_SHADOWING") val alpha = alpha.toPrimitive()
    val dRowsNum = destination.shape[0]

    val dBlocksInRow = destination.blocksInRow
    val lrBlocksInRow = this.blocksInRow

    val dBlockSize = destination.array.blockSize
    val lrBlockSize = array.blockSize

    val dBlocks = destination.array.blocks
    val lBlocks = this.array.blocks
    val rBlocks = other.array.blocks

    suspend fun wrapper(body: suspend (inner: suspend () -> Unit) -> Unit = { it() }) {
        for (dRow in 0 until dRowsNum) {
            var rRow = 0
            val dBlockOffset = dRow * dBlocksInRow
            val lBlockOffset = dRow * lrBlocksInRow

            for (dBlockInRow in 0 until dBlocksInRow) {
                val dBlock = dBlocks[dBlockOffset + dBlockInRow]
                val rRowOffset = rRow
                body {
                    var rBlockOffset = rRowOffset * lrBlocksInRow
                    for (dIdx in 0 until dBlockSize) {
                        for (lrBlockInRow in 0 until lrBlocksInRow) {
                            val lBlock = lBlocks[lBlockOffset + lrBlockInRow]
                            val rBlock = rBlocks[rBlockOffset + lrBlockInRow]

                            for (lrIdx in 0 until lrBlockSize) {
                                dBlock[dIdx] = (dBlock[dIdx] + alpha * lBlock[lrIdx] * rBlock[lrIdx]).toPrimitive()
                            }
                        }
                        rBlockOffset += lrBlocksInRow
                    }
                }

                rRow += dBlockSize
            }
        }
    }

    if (destination.blocksInRow > 1) {
        coroutineScope { wrapper { launch { it() } } }
    } else {
        wrapper()
    }

    return destination
}
