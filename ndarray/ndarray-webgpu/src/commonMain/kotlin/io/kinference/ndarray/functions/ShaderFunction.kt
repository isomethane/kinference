package io.kinference.ndarray.functions

import io.kinference.ndarray.environment.WebGPU
import io.kinference.utils.webgpu.*

abstract class ShaderFunction : NDArrayFunction {
    protected abstract val shader: String
    protected open val shaderEntryPoint: String = "main"
    protected abstract val workGroupSize: IntArray
    protected abstract val dispatchSize: IntArray
    protected abstract val bindGroupLayoutDescriptor: BindGroupLayoutDescriptor

    private var _computePipeline: ComputePipeline? = null
    private val computePipeline: ComputePipeline
        get() {
            if (_computePipeline == null) {
                _computePipeline = WebGPU.device.createComputePipeline(
                    ComputePipelineDescriptor(
                        layout = WebGPU.device.createPipelineLayout(
                            PipelineLayoutDescriptor(bindGroupLayouts = listOf(WebGPU.device.createBindGroupLayout(bindGroupLayoutDescriptor)))
                        ),
                        compute = ProgrammableStage(
                            module = WebGPU.device.createShaderModule(ShaderModuleDescriptor(shader)),
                            entryPoint = shaderEntryPoint
                        )
                    )
                )
            }
            return _computePipeline!!
        }

    protected fun performComputePass(bindGroup: BindGroup) {
        val computePass = WebGPU.beginComputePass()
        computePass.setPipeline(computePipeline)
        computePass.setBindGroup(0, bindGroup, listOf())
        computePass.dispatchWorkgroups(dispatchSize[0], dispatchSize.getOrNull(1) ?: 1, dispatchSize.getOrNull(2) ?: 1)
        computePass.end()
    }
}
