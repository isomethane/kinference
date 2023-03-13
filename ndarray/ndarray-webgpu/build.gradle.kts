import io.kinference.gradle.Versions

group = rootProject.group
version = rootProject.version

kotlin {
    jvm()

    js(BOTH) {
        browser()
    }

    sourceSets {
        val commonMain by getting {
            dependencies {
                implementation(project(":utils:webgpu-utils:webgpu-compute"))

                implementation(project(":ndarray:ndarray-core"))

                api(project(":ndarray:ndarray-api"))
                api("io.kinference.primitives:primitives-annotations:${Versions.kinferencePrimitives}")
                implementation("org.jetbrains.kotlinx:kotlinx-coroutines-core:${Versions.kotlinxCoroutines}")
            }
        }
    }
}
