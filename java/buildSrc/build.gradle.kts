// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

plugins {
    `kotlin-dsl`
}

repositories {
    gradlePluginPortal()
    mavenCentral()
}

dependencies {
    implementation("com.gradleup.shadow:shadow-gradle-plugin:9.4.2")
    implementation("com.vanniktech.maven.publish:com.vanniktech.maven.publish.gradle.plugin:0.36.0")
}
