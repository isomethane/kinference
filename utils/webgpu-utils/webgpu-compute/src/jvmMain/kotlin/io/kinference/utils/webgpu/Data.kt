package io.kinference.utils.webgpu

import io.kinference.utils.wgpu.internal.MemoryMode
import io.kinference.utils.wgpu.internal.getString
import io.kinference.utils.wgpu.jnr.*

actual class BindGroupEntry actual constructor(
    binding: Int,
    resource: BufferBinding
) : WGPUBindGroupEntry(MemoryMode.Direct) {
    init {
        this.binding = binding.toLong()
        this.buffer = resource.buffer.wgpuBuffer
        this.offset = resource.offset.toLong()
        this.size = resource.size.toLong()
    }
}

actual class BindGroupLayoutEntry actual constructor(
    binding: Int,
    buffer: BufferBindingLayout
) : WGPUBindGroupLayoutEntry(MemoryMode.Direct) {
    init {
        this.binding = binding.toLong()
        this.visibility = WGPUShaderStage.Compute.value.toLong()
        this.buffer.apply {
            type = buffer.type
            hasDynamicOffset = buffer.hasDynamicOffset
            minBindingSize = buffer.minBindingSize.toLong()
        }
    }
}

actual class BufferBinding actual constructor(
    val buffer: Buffer,
    val offset: Int,
    val size: Int
)

actual class BufferBindingLayout actual constructor(
    val type: BufferBindingType,
    val hasDynamicOffset: Boolean,
    minBindingSize: Int
) {
    val minBindingSize: Long = if (minBindingSize == 0) -1 else minBindingSize.toLong()
}

actual class BufferUsageFlags(val value: Long) {
    actual constructor(vararg flags: BufferUsage) : this(flags.map { it.value }.fold(0, Int::or).toLong())
}

actual class CompilationInfo(wgpuCompilationInfo: WGPUCompilationInfo) {
    actual val messages: List<CompilationMessage> =
        wgpuCompilationInfo.messages.get(wgpuCompilationInfo.messageCount.toInt()).map { CompilationMessage(it) }
}

actual class CompilationMessage(private val wgpuCompilationMessage: WGPUCompilationMessage) {
    actual val message: String
        get() = wgpuCompilationMessage.message.getString()
    actual val type: CompilationMessageType
        get() = wgpuCompilationMessage.type
    actual val lineNum: Int
        get() = wgpuCompilationMessage.lineNum.toInt()
    actual val linePos: Int
        get() = wgpuCompilationMessage.linePos.toInt()
    actual val offset: Int
        get() = wgpuCompilationMessage.offset.toInt()
    actual val length: Int
        get() = wgpuCompilationMessage.length.toInt()
}

actual class Limits actual constructor(
    val maxBindGroups: Long?,
    val maxBindingsPerBindGroup: Long?,
    val maxDynamicStorageBuffersPerPipelineLayout: Long?,
    val maxStorageBuffersPerShaderStage: Long?,
    val maxStorageBufferBindingSize: Long?,
    val minStorageBufferOffsetAlignment: Long?,
    val maxBufferSize: Long?,
    val maxComputeWorkgroupStorageSize: Long?,
    val maxComputeInvocationsPerWorkgroup: Long?,
    val maxComputeWorkgroupSizeX: Long?,
    val maxComputeWorkgroupSizeY: Long?,
    val maxComputeWorkgroupSizeZ: Long?,
    val maxComputeWorkgroupsPerDimension: Long?
)

actual class MapModeFlags(val value: Long) {
    actual constructor(vararg flags: MapMode) : this(flags.map { it.value }.fold(0, Int::or).toLong())
}

actual class ProgrammableStage actual constructor(
    val module: ShaderModule,
    val entryPoint: String
)

actual class RequestAdapterOptions actual constructor(
    powerPreference: PowerPreference,
    forceFallbackAdapter: kotlin.Boolean
) : WGPURequestAdapterOptions(MemoryMode.Direct) {
    init {
        this.powerPreference = powerPreference
        this.forceFallbackAdapter = forceFallbackAdapter
    }
}

actual class SupportedLimits(private val wgpuSupportedLimits: WGPUSupportedLimits) {
    actual val maxBindGroups: Long
        get() = wgpuSupportedLimits.limits.maxBindGroups
    actual val maxBindingsPerBindGroup: Long
        get() = wgpuSupportedLimits.limits.maxBindingsPerBindGroup
    actual val maxDynamicStorageBuffersPerPipelineLayout: Long
        get() = wgpuSupportedLimits.limits.maxDynamicStorageBuffersPerPipelineLayout
    actual val maxStorageBuffersPerShaderStage: Long
        get() = wgpuSupportedLimits.limits.maxStorageBuffersPerShaderStage
    actual val maxStorageBufferBindingSize: Long
        get() = wgpuSupportedLimits.limits.maxStorageBufferBindingSize
    actual val minStorageBufferOffsetAlignment: Long
        get() = wgpuSupportedLimits.limits.minStorageBufferOffsetAlignment
    actual val maxBufferSize: Long
        get() = wgpuSupportedLimits.limits.maxBufferSize
    actual val maxComputeWorkgroupStorageSize: Long
        get() = wgpuSupportedLimits.limits.maxComputeWorkgroupStorageSize
    actual val maxComputeInvocationsPerWorkgroup: Long
        get() = wgpuSupportedLimits.limits.maxComputeInvocationsPerWorkgroup
    actual val maxComputeWorkgroupSizeX: Long
        get() = wgpuSupportedLimits.limits.maxComputeWorkgroupSizeX
    actual val maxComputeWorkgroupSizeY: Long
        get() = wgpuSupportedLimits.limits.maxComputeWorkgroupSizeY
    actual val maxComputeWorkgroupSizeZ: Long
        get() = wgpuSupportedLimits.limits.maxComputeWorkgroupSizeZ
    actual val maxComputeWorkgroupsPerDimension: Long
        get() = wgpuSupportedLimits.limits.maxComputeWorkgroupsPerDimension
}
