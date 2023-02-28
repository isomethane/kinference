package io.kinference.operators.math

import io.kinference.KITestEngine.KIAccuracyRunner
import io.kinference.utils.TestRunner
import kotlin.test.Test

class MatMulIntegerTest {
    private fun getTargetPath(dirName: String) = "matmul_integer/$dirName/"

    @Test
    fun test_matmul_integer() = TestRunner.runTest {
        KIAccuracyRunner.runFromResources(getTargetPath("test_matmulinteger"))
    }

    @Test
    fun test_matmul_integer_op() = TestRunner.runTest {
        KIAccuracyRunner.runFromResources(getTargetPath("test_matmulinteger_op"))
    }
}
