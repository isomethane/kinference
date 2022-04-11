import io.kinference.gradle.optInFixed
import org.jetbrains.kotlin.gradle.dsl.KotlinJvmCompile
import org.jetbrains.kotlin.gradle.dsl.KotlinMultiplatformExtension

group = "io.kinference"
version = "0.1.14"

plugins {
    kotlin("multiplatform") apply false
    idea apply true
    id("io.gitlab.arturbosch.detekt") version ("1.18.1") apply true
    `maven-publish`
}

allprojects {
    repositories {
        maven(url = "https://packages.jetbrains.team/maven/p/ki/maven")
        mavenCentral()
        maven(url = "https://packages.jetbrains.team/maven/p/grazi/grazie-platform-public")
    }
}

subprojects {
    if (this.subprojects.isNotEmpty()) return@subprojects

    apply {
        plugin("org.jetbrains.kotlin.multiplatform")

        plugin("maven-publish")
        plugin("idea")
        plugin("io.gitlab.arturbosch.detekt")
    }


    publishing {
        repositories {
            maven {
                name = "SpacePackages"
                url = uri("https://packages.jetbrains.team/maven/p/ki/maven")

                credentials {
                    username = System.getenv("JB_SPACE_CLIENT_ID")
                    password = System.getenv("JB_SPACE_CLIENT_SECRET")
                }
            }
        }
    }

    extensions.getByType(KotlinMultiplatformExtension::class.java).apply {
        sourceSets.all {
            languageSettings {
                optInFixed("kotlin.RequiresOptIn")
                optInFixed("kotlin.time.ExperimentalTime")
                optInFixed("kotlin.ExperimentalUnsignedTypes")
                optInFixed("kotlinx.serialization.ExperimentalSerializationApi")
            }

            languageSettings {
                apiVersion = "1.6"
                languageVersion = "1.6"
            }
        }

        tasks.withType<KotlinJvmCompile> {
            kotlinOptions {
                jvmTarget = "11"
            }
        }
    }

    detekt {
        parallel = true

        config = rootProject.files("detekt.yml")

        reports {
            xml.enabled = false
            html.enabled = false
        }
    }
}
