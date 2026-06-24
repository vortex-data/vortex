// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

plugins {
    java
    id("me.champeau.jmh") version "0.7.3"
}

java {
    toolchain {
        languageVersion.set(JavaLanguageVersion.of(17))
        vendor.set(JvmVendorSpec.AMAZON)
    }
}

// This module is a benchmark harness; it is never published.
tasks.withType<AbstractPublishToMaven>().configureEach { enabled = false }

dependencies {
    jmhImplementation(platform(libs.netty.bom))
    jmhImplementation(project(":vortex-jni"))
    jmhImplementation(libs.arrow.c.data)
    jmhImplementation(libs.arrow.memory.core)
    jmhImplementation(libs.arrow.memory.netty)
}

jmh {
    jmhVersion.set("1.37")
}
