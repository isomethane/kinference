import io.kinference.gradle.configureGpuLightTests
import io.kinference.gradle.configureTests

plugins {
    kotlin("kapt") apply true
}

group = rootProject.group
version = rootProject.version

kotlin {
    js(BOTH) {
        browser()

        configureTests()
        configureGpuLightTests()
    }

    jvm {
        configureGpuLightTests()
    }

    sourceSets {
        val commonMain by getting {
            dependencies {
                api(kotlin("stdlib"))

                api(project(":ndarray:ndarray-api"))
                api(project(":ndarray:ndarray-webgpu"))
                implementation(project(":ndarray:ndarray-core"))

                api(project(":inference:inference-api"))
                api(project(":inference:inference-ir"))

                implementation(project(":utils:webgpu-utils:webgpu-compute"))
            }
        }

        val commonTest by getting {
            dependencies {
                implementation(kotlin("test"))
                implementation(kotlin("test-annotations-common"))
                implementation(project(":utils:test-utils"))
            }
        }

        val jvmTest by getting {
            dependencies {
                implementation(kotlin("test-junit5"))
                api("org.slf4j:slf4j-simple:${io.kinference.gradle.Versions.slf4j}")

                implementation("org.openjdk.jmh:jmh-core:1.25.1")
                configurations["kapt"].dependencies.add(implementation("org.openjdk.jmh:jmh-generator-annprocess:1.25.1"))
            }
        }
    }
}
