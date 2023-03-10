package io.kinference.gradle

import io.kinference.gradle.s3.S3Dependency
import org.jetbrains.kotlin.gradle.targets.js.dsl.KotlinJsTargetDsl
import org.jetbrains.kotlin.gradle.targets.js.KotlinJsTarget
import org.gradle.kotlin.dsl.*
import org.jetbrains.kotlin.gradle.targets.js.KotlinJsPlatformTestRun

fun KotlinJsPlatformTestRun.configureBrowsers() {
    executionTask.get().useKarma {
        useChrome()
    }
}

fun KotlinJsPlatformTestRun.configureTests() {
    filter {
        excludeTestsMatching("*.heavy_*")
        excludeTestsMatching("*.benchmark_*")
    }

    executionTask.get().enabled = !target.project.hasProperty("disable-tests")
}

fun KotlinJsTargetDsl.configureTests() {
    testRuns["test"].configureAllExecutions{
        configureTests()
        configureBrowsers()
    }

    (this as? KotlinJsTarget)?.irTarget?.testRuns?.get("test")?.configureAllExecutions {
        configureTests()
        executionTask.get().dependsOn(":utils:test-utils:jsLegacyProcessResources")
    }
}

fun KotlinJsPlatformTestRun.configureHeavyTests() {
    filter {
        includeTestsMatching("*.heavy_*")
    }

    executionTask.get().enabled = !target.project.hasProperty("disable-tests")
    executionTask.get().doFirst {
        S3Dependency.withDefaultS3Dependencies(this)
    }
}

fun KotlinJsTargetDsl.configureHeavyTests() {
    testRuns.create("heavy").configureAllExecutions{
        configureHeavyTests()
        configureBrowsers()
    }
    (this as KotlinJsTarget).irTarget?.testRuns?.create("heavy")?.configureAllExecutions {
        configureHeavyTests()
        executionTask.get().dependsOn(":utils:test-utils:jsLegacyProcessResources")
    }
}

fun KotlinJsPlatformTestRun.configureBenchmarkTests() {
    filter {
        includeTestsMatching("*.benchmark_*")
    }

    executionTask.get().enabled = !target.project.hasProperty("disable-tests")

    executionTask.get().doFirst {
        S3Dependency.withDefaultS3Dependencies(this)
    }
}

fun KotlinJsTargetDsl.configureBenchmarkTests() {
    testRuns.create("benchmark").configureAllExecutions {
        configureBenchmarkTests()
        configureBrowsers()
    }

    (this as KotlinJsTarget).irTarget?.testRuns?.create("benchmark")?.configureAllExecutions {
        configureBenchmarkTests()
        executionTask.get().dependsOn(":utils:test-utils:jsLegacyProcessResources")
    }
}

// TODO rename?
fun KotlinJsPlatformTestRun.configureGpuLightTests() {
    filter {
        includeTestsMatching("*.gpu_*")
    }

    executionTask.get().enabled = !target.project.hasProperty("disable-tests")
}

fun KotlinJsTargetDsl.configureGpuLightTests() {
    testRuns.create("gpu").configureAllExecutions{
        configureGpuLightTests()
        configureBrowsers()
    }

    (this as? KotlinJsTarget)?.irTarget?.testRuns?.create("gpu")?.configureAllExecutions {
        configureGpuLightTests()
        executionTask.get().dependsOn(":utils:test-utils:jsLegacyProcessResources")
    }
}
