package io.kinference.webgpu.operators.tensor

import io.kinference.utils.TestRunner
import io.kinference.webgpu.WebGPUTestEngine.WebGPUAccuracyRunner
import kotlin.test.Test

class UnsqueezeTest {
    private fun getTargetPath(dirName: String) = "/unsqueeze/$dirName/"

    @Test
    fun gpu_test_unsqueeze_axis_0() = TestRunner.runTest {
        WebGPUAccuracyRunner.runFromResources(getTargetPath("test_unsqueeze_axis_0"))
    }

    @Test
    fun gpu_test_unsqueeze_axis_1() = TestRunner.runTest {
        WebGPUAccuracyRunner.runFromResources(getTargetPath("test_unsqueeze_axis_1"))
    }

    @Test
    fun gpu_test_unsqueeze_axis_2() = TestRunner.runTest {
        WebGPUAccuracyRunner.runFromResources(getTargetPath("test_unsqueeze_axis_2"))
    }

    @Test
    fun gpu_test_unsqueeze_axis_3() = TestRunner.runTest {
        WebGPUAccuracyRunner.runFromResources(getTargetPath("test_unsqueeze_axis_3"))
    }

    @Test
    fun gpu_test_unsqueeze_with_negative_axes() = TestRunner.runTest {
        WebGPUAccuracyRunner.runFromResources(getTargetPath("test_unsqueeze_negative_axes"))
    }

    @Test
    fun gpu_test_unsqueeze_three_axes() = TestRunner.runTest {
        WebGPUAccuracyRunner.runFromResources(getTargetPath("test_unsqueeze_three_axes"))
    }

    @Test
    fun gpu_test_unsqueeze_two_axes() = TestRunner.runTest {
        WebGPUAccuracyRunner.runFromResources(getTargetPath("test_unsqueeze_two_axes"))
    }

    @Test
    fun gpu_test_unsqueeze_unsorted_axes() = TestRunner.runTest {
        WebGPUAccuracyRunner.runFromResources(getTargetPath("test_unsqueeze_unsorted_axes"))
    }
}
