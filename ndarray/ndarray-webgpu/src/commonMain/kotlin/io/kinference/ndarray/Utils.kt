package io.kinference.ndarray

import io.kinference.ndarray.arrays.*
import io.kinference.utils.webgpu.BufferData

fun BufferData.unpack(info: NDArrayInfo): TypedNDArrayData = when (info.type) {
    WebGPUDataType.INT32 -> IntNDArrayData(this.toIntArray())
    WebGPUDataType.UINT32 -> UIntNDArrayData(this.toUIntArray())
    WebGPUDataType.FLOAT32 -> FloatNDArrayData(this.toFloatArray())
}

fun BooleanArray.toIntArray() = IntArray(this.size) { if (this[it]) 1 else 0 }
fun ByteArray.toIntArray() = IntArray(this.size) { this[it].toInt() }
fun ShortArray.toIntArray() = IntArray(this.size) { this[it].toInt() }
fun IntArray.toByteArray() = ByteArray(this.size) { this[it].toByte() }
fun IntArray.toUByteArray() = UByteArray(this.size) { this[it].toUByte() }
fun IntArray.toBooleanArray() = BooleanArray(this.size) { this[it] != 0 }
fun IntArray.toLongArray() = LongArray(this.size) { this[it].toLong() }
fun LongArray.toIntArray() = IntArray(this.size) { this[it].toInt() }

fun UByteArray.toUIntArray() = UIntArray(this.size) { this[it].toUInt() }
fun UShortArray.toUIntArray() = UIntArray(this.size) { this[it].toUInt() }
fun ULongArray.toUIntArray() = UIntArray(this.size) { this[it].toUInt() }

fun DoubleArray.toFloatArray() = FloatArray(this.size) { this[it].toFloat() }
