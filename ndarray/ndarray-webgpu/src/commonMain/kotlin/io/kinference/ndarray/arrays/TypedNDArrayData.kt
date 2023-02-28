package io.kinference.ndarray.arrays

sealed class TypedNDArrayData
class IntNDArrayData(val data: IntArray) : TypedNDArrayData()
class UIntNDArrayData(val data: UIntArray) : TypedNDArrayData()
class FloatNDArrayData(val data: FloatArray) : TypedNDArrayData()
