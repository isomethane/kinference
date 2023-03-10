package io.kinference.utils.webgpu

import kotlin.js.Json
import kotlin.js.json

actual class BindGroupEntry actual constructor(
    override var binding: Int,
    resource: BufferBinding
): GPUBindGroupEntry {
    override var resource: GPUBufferBinding = resource
}
external interface GPUBindGroupEntry {
    var binding: Int
    var resource: GPUBufferBinding
}

actual class BindGroupLayoutEntry actual constructor(
    override var binding: Int,
    buffer: BufferBindingLayout
): GPUBindGroupLayoutEntry {
    override var buffer: GPUBufferBindingLayout = buffer
    override var visibility: Int = 0x4 // COMPUTE
}
external interface GPUBindGroupLayoutEntry {
    var binding: Int
    var buffer: GPUBufferBindingLayout
    var visibility: Int
}

actual class BufferBinding actual constructor(
    buffer: Buffer,
    override var offset: Int,
    override var size: Int
): GPUBufferBinding {
    override var buffer: GPUBuffer = buffer.gpuBuffer
}
external interface GPUBufferBinding {
    var buffer: GPUBuffer
    var offset: Int
    var size: Int
}

actual class BufferBindingLayout actual constructor(
    type: BufferBindingType,
    hasDynamicOffset: Boolean,
    override var minBindingSize: Int
): GPUBufferBindingLayout {
    override var type: String = type.value
    override var hasDynamicOffset: Boolean? = hasDynamicOffset
}
external interface GPUBufferBindingLayout {
    var type: String
    var hasDynamicOffset: Boolean?
    var minBindingSize: Int
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
    override var entryPoint: String
): GPUProgrammableStage {
    override var module = module.gpuShaderModule
}
external interface GPUProgrammableStage {
    var module: GPUShaderModule
    var entryPoint: String
}

actual class RequestAdapterOptions actual constructor(
    powerPreference: PowerPreference,
    forceFallbackAdapter: Boolean
): GPURequestAdapterOptions {
    override var powerPreference: String = powerPreference.value
    override var forceFallbackAdapter: Boolean? = forceFallbackAdapter
}
external interface GPURequestAdapterOptions {
    var powerPreference: String
    var forceFallbackAdapter: Boolean?
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
