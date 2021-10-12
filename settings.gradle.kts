rootProject.name = "kinference"

include(":ndarray")
include(":serialization")
include(":inference-api")
include(":inference")
include(":inference-ort")

include(":utils:test-utils")
include(":utils:logger")
include(":utils:model-profiler")

include(":adapters:adapter-multik")
include(":adapters:adapter-kmath")


pluginManagement {
    repositories {
        gradlePluginPortal()
        mavenCentral()
        maven(url = "https://packages.jetbrains.team/maven/p/ki/maven")
    }

    resolutionStrategy {
        eachPlugin {
            if (requested.id.id == "io.kinference.primitives") {
                useModule("io.kinference.primitives:gradle-plugin-jvm:${requested.version}")
            }
        }
    }
}
