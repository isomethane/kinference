import io.kinference.gradle.Versions

group = rootProject.group
version = rootProject.version

plugins {
    id("io.kinference.primitives") version "0.1.19" apply true
}

kotlin {
    jvm()

    js(BOTH) {
        browser()
    }

    sourceSets {
        val commonMain by getting {
            dependencies {
                api(project(":ndarray:ndarray-api"))
                api("io.kinference.primitives:primitives-annotations:${Versions.kinferencePrimitives}")
                implementation("org.jetbrains.kotlinx:kotlinx-coroutines-core:${Versions.kotlinxCoroutines}")
            }
        }
    }
}
