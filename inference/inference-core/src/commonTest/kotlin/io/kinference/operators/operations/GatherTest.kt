package io.kinference.operators.operations

import io.kinference.KITestEngine.KIAccuracyRunner
import io.kinference.utils.TestRunner
import kotlin.test.Test

class GatherTest {
    private fun getTargetPath(dirName: String) = "gather/$dirName/"

    @Test
    fun test_gather_0() = TestRunner.runTest {
        KIAccuracyRunner.runFromResources(getTargetPath("test_gather_0"))
    }

    @Test
    fun test_gather_1() = TestRunner.runTest {
        KIAccuracyRunner.runFromResources(getTargetPath("test_gather_1"))
    }

    @Test
    fun test_gather_with_negative_indices() = TestRunner.runTest {
        KIAccuracyRunner.runFromResources(getTargetPath("test_gather_negative_indices"))
    }
}

