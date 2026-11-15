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

val sparkModules =
    mapOf(
        "3.5" to listOf("2.12", "2.13"),
        "4.0" to listOf("2.13"),
    )
val requestedSparkVersions =
    System
        .getProperty("sparkVersions")
        ?.split(',')
        ?.map(String::trim)
        ?.filter(String::isNotEmpty)
        ?.toSet()

sparkModules.forEach { (sparkVersion, scalaVersions) ->
    if (requestedSparkVersions == null || sparkVersion in requestedSparkVersions) {
        scalaVersions.forEach { scalaVersion ->
            val projectName = "vortex-spark-${sparkVersion}_$scalaVersion"
            include(projectName)
            project(":$projectName").projectDir = file("vortex-spark/v$sparkVersion")
        }
    }
}
