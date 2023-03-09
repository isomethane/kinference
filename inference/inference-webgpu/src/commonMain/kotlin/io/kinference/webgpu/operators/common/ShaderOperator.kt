package io.kinference.webgpu.operators.common

import io.kinference.attribute.Attribute
import io.kinference.ndarray.WebGPUState
import io.kinference.operator.Operator
import io.kinference.operator.OperatorInfo
import io.kinference.utils.webgpu.*
import io.kinference.webgpu.data.tensor.WebGPUTensor
import io.kinference.webgpu.engine.WebGPUEnvironment

abstract class ShaderOperator(
    name: String, info: OperatorInfo, attributes: Map<String, Attribute<Any>>, inputs: List<String>, outputs: List<String>
) : Operator<WebGPUTensor, WebGPUTensor>(name, info, attributes, inputs, outputs) {
    protected abstract val shader: String
    protected open val shaderEntryPoint: String = "main"
    protected abstract val workGroupSize: IntArray
    protected abstract val dispatchSize: IntArray
    protected abstract val bindGroupLayoutDescriptor: BindGroupLayoutDescriptor

    private var _computePipeline: ComputePipeline? = null
    private val computePipeline: ComputePipeline
        get() {
            if (_computePipeline == null) {
                _computePipeline = WebGPUEnvironment.gpuState.device.createComputePipeline(
                    ComputePipelineDescriptor(
                        layout = WebGPUEnvironment.gpuState.device.createPipelineLayout(
                            PipelineLayoutDescriptor(bindGroupLayouts = listOf(WebGPUEnvironment.gpuState.device.createBindGroupLayout(bindGroupLayoutDescriptor)))
                        ),
                        compute = ProgrammableStage(
                            module = WebGPUEnvironment.gpuState.device.createShaderModule(ShaderModuleDescriptor(shader)),
                            entryPoint = shaderEntryPoint
                        )
                    )
                )
            }
            return _computePipeline!!
        }

    protected fun performComputePass(gpuState: WebGPUState, bindGroup: BindGroup) {
        val computePass = gpuState.beginComputePass()
        computePass.setPipeline(computePipeline)
        computePass.setBindGroup(0, bindGroup, listOf())
        computePass.dispatchWorkgroups(dispatchSize[0], dispatchSize.getOrNull(1) ?: 1, dispatchSize.getOrNull(2) ?: 1)
        computePass.end()
    }
}
