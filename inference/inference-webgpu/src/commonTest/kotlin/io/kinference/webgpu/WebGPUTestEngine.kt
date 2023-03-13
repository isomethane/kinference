package io.kinference.webgpu

import io.kinference.TestEngine
import io.kinference.runners.AccuracyRunner
import io.kinference.runners.PerformanceRunner
import io.kinference.webgpu.utils.WebGPUAssertions

object WebGPUTestEngine : TestEngine<WebGPUData<*>>(WebGPUEngine) {
    override fun checkEquals(expected: WebGPUData<*>, actual: WebGPUData<*>, delta: Double) {
        WebGPUAssertions.assertEquals(expected, actual, delta)
    }

    val WebGPUAccuracyRunner = AccuracyRunner(WebGPUTestEngine)

    val WebGPUPerformanceRunner = PerformanceRunner(WebGPUTestEngine)
}
