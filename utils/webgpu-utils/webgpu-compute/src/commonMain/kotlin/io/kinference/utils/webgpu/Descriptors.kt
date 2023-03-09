package io.kinference.utils.webgpu

expect class BindGroupDescriptor(
    layout: BindGroupLayout,
    entries: List<BindGroupEntry>,
)

expect class BindGroupLayoutDescriptor(
    entries: List<BindGroupLayoutEntry>
)

expect class BufferDescriptor(
    size: Int,
    usage: BufferUsageFlags,
    mappedAtCreation: Boolean = false
)

expect class CommandBufferDescriptor()

expect class CommandEncoderDescriptor()

expect class ComputePassDescriptor()

expect class ComputePipelineDescriptor(
    layout: PipelineLayout? = null,
    compute: ProgrammableStage,
)

expect class DeviceDescriptor(
    requiredLimits: Limits
)

expect class InstanceDescriptor()

expect class PipelineLayoutDescriptor(
    bindGroupLayouts: List<BindGroupLayout>,
)

expect class ShaderModuleDescriptor(
    code: String,
)
