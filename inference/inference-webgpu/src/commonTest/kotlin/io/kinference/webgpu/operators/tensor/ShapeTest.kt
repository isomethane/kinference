package io.kinference.webgpu.operators.tensor

import io.kinference.utils.TestRunner
import io.kinference.webgpu.WebGPUTestEngine.WebGPUAccuracyRunner
import kotlin.test.Test

class ShapeTest {
    private fun getTargetPath(dirName: String) = "shape/$dirName/"

    @Test
    fun gpu_test_shape() = TestRunner.runTest {
        WebGPUAccuracyRunner.runFromResources(getTargetPath("test_shape"))
    }

    @Test
    fun gpu_test_shape_example() = TestRunner.runTest {
        WebGPUAccuracyRunner.runFromResources(getTargetPath("test_shape_example"))
    }
}
