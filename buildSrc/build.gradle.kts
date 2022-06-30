repositories {
    mavenCentral()
}

plugins {
    `kotlin-dsl`
}

dependencies {
    implementation("org.jetbrains.kotlin:kotlin-gradle-plugin:1.7.0")
    implementation(gradleApi())
    implementation(gradleKotlinDsl())
    api("com.amazonaws:aws-java-sdk-s3:1.11.896")
}
