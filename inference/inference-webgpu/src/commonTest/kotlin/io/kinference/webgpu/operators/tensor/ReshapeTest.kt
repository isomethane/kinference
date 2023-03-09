package io.kinference.webgpu.operators.tensor

import io.kinference.utils.TestRunner
import io.kinference.webgpu.WebGPUTestEngine.WebGPUAccuracyRunner
import kotlin.test.Test

class ReshapeTest {
    private fun getTargetPath(dirName: String) = "reshape/$dirName/"

    @Test
    fun gpu_test_reshape_with_extended_dimensions() = TestRunner.runTest {
        WebGPUAccuracyRunner.runFromResources(getTargetPath("test_reshape_extended_dims"))
    }

    @Test
    fun gpu_test_reshape_with_negative_dimension() = TestRunner.runTest {
        WebGPUAccuracyRunner.runFromResources(getTargetPath("test_reshape_negative_dim"))
    }

    @Test
    fun gpu_test_reshape_with_negative_extended_dimensions() = TestRunner.runTest {
        WebGPUAccuracyRunner.runFromResources(getTargetPath("test_reshape_negative_extended_dims"))
    }

    @Test
    fun gpu_test_reshape_with_one_dimension() = TestRunner.runTest {
        WebGPUAccuracyRunner.runFromResources(getTargetPath("test_reshape_one_dim"))
    }

    @Test
    fun gpu_test_reshape_with_reduced_dimensions() = TestRunner.runTest {
        WebGPUAccuracyRunner.runFromResources(getTargetPath("test_reshape_reduced_dims"))
    }

    @Test
    fun gpu_test_reshape_with_all_reordered_dimensions() = TestRunner.runTest {
        WebGPUAccuracyRunner.runFromResources(getTargetPath("test_reshape_reordered_all_dims"))
    }

    @Test
    fun gpu_test_reshape_with_reordered_last_dimensions() = TestRunner.runTest {
        WebGPUAccuracyRunner.runFromResources(getTargetPath("test_reshape_reordered_last_dims"))
    }

    @Test
    fun gpu_test_reshape_with_zero_and_negative_dimension() = TestRunner.runTest {
        WebGPUAccuracyRunner.runFromResources(getTargetPath("test_reshape_zero_and_negative_dim"))
    }

    @Test
    fun gpu_test_reshape_zero_dim() = TestRunner.runTest {
        WebGPUAccuracyRunner.runFromResources(getTargetPath("test_reshape_zero_dim"))
    }
}
