package io.kinference.models

import io.kinference.runners.KITestEngine.KIAccuracyRunner
import io.kinference.runners.PerformanceRunner
import io.kinference.utils.TestRunner
import kotlin.test.Test
import kotlin.time.ExperimentalTime

@ExperimentalTime
class POSTest {
    @Test
    fun heavy_test_pos_tagger() = TestRunner.runTest {
        KIAccuracyRunner.runFromResources("/pos_tagger/")
    }

    @Test
    fun benchmark_test_pos_tagger_performance() = TestRunner.runTest {
        PerformanceRunner.runFromResources("/pos_tagger/")
    }
}
