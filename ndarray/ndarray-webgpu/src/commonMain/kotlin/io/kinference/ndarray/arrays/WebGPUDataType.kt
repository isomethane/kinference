package io.kinference.ndarray.arrays

enum class WebGPUDataType(val wgslType: String, val sizeBytes: Int) {
    INT32("i32", 4),
    UINT32("u32", 4),
    FLOAT32("f32", 4);
}
