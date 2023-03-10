package io.kinference.utils.webgpu

import kotlin.math.max

expect class BindGroupEntry(
    binding: Int,
    resource: BufferBinding,
)

expect class BindGroupLayoutEntry(
    binding: Int,
    buffer: BufferBindingLayout,
)

expect class BufferBinding(
    buffer: Buffer,
    offset: Int = 0,
    size: Int = max(0, buffer.size() - offset),
)

expect class BufferBindingLayout(
    type: BufferBindingType,
    hasDynamicOffset: Boolean = false,
    minBindingSize: Int = 0
)

expect class BufferUsageFlags(
    vararg flags: BufferUsage
)

expect class CompilationInfo {
    val messages: List<CompilationMessage>
}

expect class CompilationMessage {
    val message: String
    val type: CompilationMessageType
    val lineNum: Int
    val linePos: Int
    val offset: Int
    val length: Int
}

expect class Limits(
    maxBindGroups: Long? = null,
    maxBindingsPerBindGroup: Long? = null,
    maxDynamicStorageBuffersPerPipelineLayout: Long? = null,
    maxStorageBuffersPerShaderStage: Long? = null,
    maxStorageBufferBindingSize: Long? = null,
    minStorageBufferOffsetAlignment: Long? = null,
    maxBufferSize: Long? = null,
    maxComputeWorkgroupStorageSize: Long? = null,
    maxComputeInvocationsPerWorkgroup: Long? = null,
    maxComputeWorkgroupSizeX: Long? = null,
    maxComputeWorkgroupSizeY: Long? = null,
    maxComputeWorkgroupSizeZ: Long? = null,
    maxComputeWorkgroupsPerDimension: Long? = null,
)

expect class MapModeFlags(
    vararg flags: MapMode,
)

expect class ProgrammableStage(
    module: ShaderModule,
    entryPoint: String,
)

expect class RequestAdapterOptions(
    powerPreference: PowerPreference = PowerPreference.HighPerformance,
    forceFallbackAdapter: Boolean = false,
)

expect class SupportedLimits {
    val maxBindGroups: Long
    val maxBindingsPerBindGroup: Long
    val maxDynamicStorageBuffersPerPipelineLayout: Long
    val maxStorageBuffersPerShaderStage: Long
    val maxStorageBufferBindingSize: Long
    val minStorageBufferOffsetAlignment: Long
    val maxBufferSize: Long
    val maxComputeWorkgroupStorageSize: Long
    val maxComputeInvocationsPerWorkgroup: Long
    val maxComputeWorkgroupSizeX: Long
    val maxComputeWorkgroupSizeY: Long
    val maxComputeWorkgroupSizeZ: Long
    val maxComputeWorkgroupsPerDimension: Long
}

fun SupportedLimits.asString() =
    """
        |{
        |   maxBindGroups: $maxBindGroups,
        |   maxBindingsPerBindGroup: $maxBindingsPerBindGroup,
        |   maxDynamicStorageBuffersPerPipelineLayout: $maxDynamicStorageBuffersPerPipelineLayout,
        |   maxStorageBuffersPerShaderStage: $maxStorageBuffersPerShaderStage,
        |   maxStorageBufferBindingSize: $maxStorageBufferBindingSize,
        |   minStorageBufferOffsetAlignment: $minStorageBufferOffsetAlignment,
        |   maxBufferSize: $maxBufferSize,
        |   maxComputeWorkgroupStorageSize: $maxComputeWorkgroupStorageSize,
        |   maxComputeInvocationsPerWorkgroup: $maxComputeInvocationsPerWorkgroup,
        |   maxComputeWorkgroupSizeX: $maxComputeWorkgroupSizeX,
        |   maxComputeWorkgroupSizeY: $maxComputeWorkgroupSizeY,
        |   maxComputeWorkgroupSizeZ: $maxComputeWorkgroupSizeZ,
        |   maxComputeWorkgroupsPerDimension: $maxComputeWorkgroupsPerDimension,
        |}
    """.trimMargin()
