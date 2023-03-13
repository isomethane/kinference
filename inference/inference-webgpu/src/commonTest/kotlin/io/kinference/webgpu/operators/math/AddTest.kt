package io.kinference.webgpu.operators.math

import io.kinference.utils.TestRunner
import io.kinference.webgpu.WebGPUTestEngine.WebGPUAccuracyRunner
import kotlin.test.Test

class AddTest {
    private fun getTargetPath(dirName: String) = "add/$dirName/"

    @Test
    fun gpu_test_add() = TestRunner.runTest {
        WebGPUAccuracyRunner.runFromResources(getTargetPath("test_add"))
    }

    @Test
    fun gpu_test_add_broadcast() = TestRunner.runTest {
        WebGPUAccuracyRunner.runFromResources(getTargetPath("test_add_bcast"))
    }

    @Test
    fun gpu_test_add_scalar() = TestRunner.runTest {
        WebGPUAccuracyRunner.runFromResources(getTargetPath("test_add_scalar"))
    }
}
