package io.kinference.webgpu.operators.math

import io.kinference.utils.TestRunner
import io.kinference.webgpu.WebGPUTestEngine.WebGPUAccuracyRunner
import kotlin.test.Test

class MulTest {
    private fun getTargetPath(dirName: String) = "mul/$dirName/"

    @Test
    fun gpu_test_mul() = TestRunner.runTest {
        WebGPUAccuracyRunner.runFromResources(getTargetPath("test_mul"))
    }

    @Test
    fun gpu_test_mul_with_broadcast() = TestRunner.runTest {
        WebGPUAccuracyRunner.runFromResources(getTargetPath("test_mul_bcast"))
    }

    @Test
    fun gpu_test_mul_defaults() = TestRunner.runTest {
        WebGPUAccuracyRunner.runFromResources(getTargetPath("test_mul_example"))
    }
}
