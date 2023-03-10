package io.kinference.utils.webgpu

import kotlin.js.Json

actual class BindGroupDescriptor actual constructor(
    override var layout: BindGroupLayout,
    entries: List<BindGroupEntry>,
): GPUBindGroupDescriptor {
    override var entries: Array<GPUBindGroupEntry> = entries.toTypedArray()
}
external interface GPUBindGroupDescriptor {
    var layout: GPUBindGroupLayout
    var entries: Array<GPUBindGroupEntry>
}

actual class BindGroupLayoutDescriptor actual constructor(
    entries: List<BindGroupLayoutEntry>,
): GPUBindGroupLayoutDescriptor {
    override var entries: Array<GPUBindGroupLayoutEntry> = entries.toTypedArray()
}
external interface GPUBindGroupLayoutDescriptor {
    var entries: Array<GPUBindGroupLayoutEntry>
}

actual class BufferDescriptor actual constructor(
    override var size: Int,
    usage: BufferUsageFlags,
    mappedAtCreation: Boolean,
): GPUBufferDescriptor {
    override var usage: Long = usage.value
    override var mappedAtCreation: Boolean? = mappedAtCreation
}
external interface GPUBufferDescriptor {
    var size: Int
    var usage: Long
    var mappedAtCreation: Boolean?
}

actual class CommandBufferDescriptor

actual class CommandEncoderDescriptor

actual class ComputePassDescriptor

actual class ComputePipelineDescriptor actual constructor(
    layout: PipelineLayout?,
    compute: ProgrammableStage,
): GPUComputePipelineDescriptor {
    override var layout: Any = layout ?: "auto"
    override var compute: GPUProgrammableStage = compute
}
external interface GPUComputePipelineDescriptor {
    var layout: Any
    var compute: GPUProgrammableStage
}

actual class DeviceDescriptor actual constructor(requiredLimits: Limits): GPUDeviceDescriptor {
    override var requiredLimits: Json = requiredLimits.record
}
external interface GPUDeviceDescriptor {
    var requiredLimits: Json
}

actual class InstanceDescriptor

actual class PipelineLayoutDescriptor actual constructor(
    bindGroupLayouts: List<BindGroupLayout>,
): GPUPipelineLayoutDescriptor {
    override var bindGroupLayouts = bindGroupLayouts.toTypedArray()
}
external interface GPUPipelineLayoutDescriptor {
    var bindGroupLayouts: Array<GPUBindGroupLayout>
}

actual class ShaderModuleDescriptor actual constructor(
    override var code: String,
): GPUShaderModuleDescriptor
external interface GPUShaderModuleDescriptor {
    var code: String
}
