package io.kinference.operators.activations

import io.kinference.KITestEngine.KIAccuracyRunner
import io.kinference.utils.TestRunner
import kotlin.test.Test
import kotlin.time.ExperimentalTime

@OptIn(ExperimentalTime::class)
class LogTest {
    private fun getTargetPath(dirName: String) = "log/$dirName/"

    @Test
    fun test_log() = TestRunner.runTest {
        KIAccuracyRunner.runFromResources(getTargetPath("test_log"))
    }

    @Test
    fun test_log_example() = TestRunner.runTest {
        KIAccuracyRunner.runFromResources(getTargetPath("test_log_example"))
    }
}
