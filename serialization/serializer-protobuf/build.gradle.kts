import io.kinference.gradle.Versions

group = rootProject.group
version = rootProject.version

kotlin {
    js(BOTH) {
        browser()
    }

    jvm()

    sourceSets {
        val commonMain by getting {
            dependencies {
                api("io.kinference.primitives:primitives-annotations:${Versions.kinferencePrimitives}")
                api("com.squareup.wire:wire-runtime:${Versions.wire}")
            }
        }
    }
}
