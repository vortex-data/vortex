// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

plugins {
    id("org.gradle.toolchains.foojay-resolver") version "1.0.0"
}

toolchainManagement {
    jvm {
        javaRepositories {
            repository("amazon-corretto") {
                resolverClass.set(org.gradle.toolchains.foojay.FoojayToolchainResolver::class.java)
            }
        }
    }
}

rootProject.name = "vortex-root"

// API bindings
include("vortex-jni")
include("vortex-spark_2.12")
project(":vortex-spark_2.12").projectDir = file("vortex-spark")

include("vortex-spark_2.13")
project(":vortex-spark_2.13").projectDir = file("vortex-spark")
