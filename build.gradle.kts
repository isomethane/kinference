import org.jetbrains.kotlin.gradle.dsl.KotlinJvmCompile
import org.jetbrains.kotlin.gradle.dsl.KotlinMultiplatformExtension

group = "io.kinference"
version = "0.2.6"

plugins {
    kotlin("multiplatform") apply false
    idea apply true
    id("io.gitlab.arturbosch.detekt") version ("1.20.0-RC2") apply true
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
        when {
            name.endsWith("-jvm") -> plugin("org.jetbrains.kotlin.jvm")
            else -> plugin("org.jetbrains.kotlin.multiplatform")
        }

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

    extensions.findByType(KotlinMultiplatformExtension::class.java)?.apply {
        sourceSets.all {
            languageSettings {
                optIn("kotlin.RequiresOptIn")
                optIn("kotlin.time.ExperimentalTime")
                optIn("kotlin.ExperimentalUnsignedTypes")
                optIn("kotlinx.serialization.ExperimentalSerializationApi")
            }

            languageSettings {
                apiVersion = "1.7"
                languageVersion = "1.7"
            }
        }
    }

    tasks.withType<KotlinJvmCompile> {
        kotlinOptions {
            jvmTarget = "17"
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
