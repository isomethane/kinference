package io.kinference.utils.webgpu

import kotlin.js.Json
import kotlin.js.json

actual class BindGroupEntry actual constructor(
    val binding: Int,
    val resource: BufferBinding
)

actual class BindGroupLayoutEntry actual constructor(
    val binding: Int,
    val buffer: BufferBindingLayout
) {
    val visibility = 0x4 // COMPUTE
}

actual class BufferBinding actual constructor(
    buffer: Buffer,
    val offset: Int,
    val size: Int
) {
    val buffer = buffer.gpuBuffer
}

actual class BufferBindingLayout actual constructor(
    type: BufferBindingType,
    val hasDynamicOffset: Boolean,
    val minBindingSize: Int
) {
    val type = type.value
}

actual class BufferUsageFlags(val value: GPUBufferUsageFlags) {
    actual constructor(vararg flags: BufferUsage) : this(flags.map { it.value }.fold(0L, Long::or))
}
typealias GPUBufferUsageFlags = Long

actual class CompilationInfo(gpuCompilationInfo: GPUCompilationInfo) {
    actual val messages: List<CompilationMessage> = gpuCompilationInfo.messages.map { CompilationMessage(it) }
}
external class GPUCompilationInfo {
    val messages: Array<GPUCompilationMessage>
}

actual class CompilationMessage(gpuCompilationMessage: GPUCompilationMessage) {
    actual val message: String = gpuCompilationMessage.message
    actual val type: CompilationMessageType = CompilationMessageType.values().single { it.value == gpuCompilationMessage.type }
    actual val lineNum: Int = gpuCompilationMessage.lineNum
    actual val linePos: Int = gpuCompilationMessage.linePos
    actual val offset: Int = gpuCompilationMessage.offset
    actual val length: Int = gpuCompilationMessage.length
}
external class GPUCompilationMessage {
    val message: String
    val type: GPUCompilationMessageType
    val lineNum: Int
    val linePos: Int
    val offset: Int
    val length: Int
}

actual class Limits(val record: Json) {
    actual constructor(
        maxBindGroups: Long?,
        maxBindingsPerBindGroup: Long?,
        maxDynamicStorageBuffersPerPipelineLayout: Long?,
        maxStorageBuffersPerShaderStage: Long?,
        maxStorageBufferBindingSize: Long?,
        minStorageBufferOffsetAlignment: Long?,
        maxBufferSize: Long?,
        maxComputeWorkgroupStorageSize: Long?,
        maxComputeInvocationsPerWorkgroup: Long?,
        maxComputeWorkgroupSizeX: Long?,
        maxComputeWorkgroupSizeY: Long?,
        maxComputeWorkgroupSizeZ: Long?,
        maxComputeWorkgroupsPerDimension: Long?
    ) : this(
        json(
            *listOf(
                "maxBindGroups" to maxBindGroups,
                "maxBindingsPerBindGroup" to maxBindingsPerBindGroup,
                "maxDynamicStorageBuffersPerPipelineLayout" to maxDynamicStorageBuffersPerPipelineLayout,
                "maxStorageBuffersPerShaderStage" to maxStorageBuffersPerShaderStage,
                "maxStorageBufferBindingSize" to maxStorageBufferBindingSize,
                "minStorageBufferOffsetAlignment" to minStorageBufferOffsetAlignment,
                "maxBufferSize" to maxBufferSize,
                "maxComputeWorkgroupStorageSize" to maxComputeWorkgroupStorageSize,
                "maxComputeInvocationsPerWorkgroup" to maxComputeInvocationsPerWorkgroup,
                "maxComputeWorkgroupSizeX" to maxComputeWorkgroupSizeX,
                "maxComputeWorkgroupSizeY" to maxComputeWorkgroupSizeY,
                "maxComputeWorkgroupSizeZ" to maxComputeWorkgroupSizeZ,
                "maxComputeWorkgroupsPerDimension" to maxComputeWorkgroupsPerDimension,
            ).filter { (_, value) ->
                value != null
            }.toTypedArray()
        )
    )
}

actual class MapModeFlags(val value: GPUMapModeFlags) {
    actual constructor(vararg flags: MapMode) : this(flags.map { it.value }.fold(0L, Long::or))
}
typealias GPUMapModeFlags = Long

actual class ProgrammableStage actual constructor(
    module: ShaderModule,
    val entryPoint: String
) {
    val module = module.gpuShaderModule
}

actual class RequestAdapterOptions actual constructor(
    powerPreference: PowerPreference,
    val forceFallbackAdapter: Boolean
) {
    val powerPreference = powerPreference.value
}

actual typealias SupportedLimits = GPUSupportedLimits
external class GPUSupportedLimits {
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
