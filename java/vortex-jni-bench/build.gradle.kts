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

// Guard: the benchmark is meaningless against a debug native lib. Require the deliberate release path
// (VORTEX_SKIP_MAKE_TEST_FILES=true, with a release libvortex_jni placed in vortex-jni's resources) so a
// plain `./gradlew :vortex-jni-bench:jmh` cannot silently rebuild + measure the debug lib.
tasks.named("jmh") {
    doFirst {
        if (System.getenv("VORTEX_SKIP_MAKE_TEST_FILES") != "true") {
            throw GradleException(
                "vortex-jni-bench must run against a RELEASE native lib. Build it " +
                    "(cargo build --release -p vortex-jni), copy it into " +
                    "vortex-jni/src/main/resources/native/<os>-<arch>/, and re-run with " +
                    "VORTEX_SKIP_MAKE_TEST_FILES=true. See README.md.",
            )
        }
    }
}
