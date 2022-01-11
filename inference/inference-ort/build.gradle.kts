import io.kinference.gradle.configureBenchmarkTests
import io.kinference.gradle.configureHeavyTests
import io.kinference.gradle.Versions

group = rootProject.group
version = rootProject.version

kotlin {
    jvm {
        testRuns.create("heavy").executionTask {
            configureHeavyTests()

            enabled = !project.hasProperty("disable-tests")
        }

        testRuns.create("benchmark").executionTask {
            configureBenchmarkTests()

            enabled = !project.hasProperty("disable-tests")
        }
    }

    sourceSets {
        val commonMain by getting {
            dependencies {
                api(project(":inference:inference-api"))
                api(project(":serialization"))
                api(project(":utils:logger"))
            }
        }


        val commonTest by getting {
            dependencies {
                implementation(kotlin("test"))
                implementation(kotlin("test-annotations-common"))
                implementation(project(":utils:test-utils"))
            }
        }

        val jvmMain by getting {
            dependencies {
                api("com.microsoft.onnxruntime:onnxruntime:${Versions.ONNXRuntime}")
            }
        }

        val jvmTest by getting {
            dependencies {
                implementation(kotlin("test-junit5"))
                api("org.slf4j:slf4j-simple:${Versions.slf4j}")
                runtimeOnly("org.junit.jupiter:junit-jupiter-engine:5.6.2")
            }
        }
    }
}

idea {
    module.generatedSourceDirs.plusAssign(files("src/commonMain/kotlin-gen"))
}
