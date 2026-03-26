// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

import com.github.jengelman.gradle.plugins.shadow.tasks.ShadowJar

plugins {
    `java-library`
    `jvm-test-suite`
    id("com.gradleup.shadow") version "9.4.0"
}

// Derive Scala and Spark versions from the Gradle project name (vortex-spark_2.12 or vortex-spark_2.13)
val scalaVersion: String = project.name.substringAfterLast("_")
val sparkVersion: String =
    when (scalaVersion) {
        "2.12" -> {
            libs.versions.spark3.get()
        }

        "2.13" -> {
            libs.versions.spark4.get()
        }

        else -> {
            throw GradleException(
                "Unsupported Scala version: $scalaVersion (project name must end with _2.12 or _2.13)",
            )
        }
    }

// Both vortex-spark_2.12 and vortex-spark_2.13 share this projectDir.
// Give each its own build directory to avoid output conflicts.
layout.buildDirectory = layout.projectDirectory.dir("build/${project.name}")

dependencies {
    compileOnly("org.apache.spark:spark-catalyst_$scalaVersion:$sparkVersion")
    compileOnly("org.apache.spark:spark-sql_$scalaVersion:$sparkVersion")
    api(project(":vortex-jni", configuration = "shadow"))

    compileOnly(libs.immutables.value)
    annotationProcessor(libs.immutables.value)

    implementation(libs.guava)
    implementation(libs.slf4j.api)
}

testing {
    suites {
        val test by getting(JvmTestSuite::class) {
            useJUnitJupiter()

            dependencies {
                implementation(libs.junit.jupiter)
                implementation("org.apache.spark:spark-core_$scalaVersion:$sparkVersion")
                implementation("org.apache.spark:spark-sql_$scalaVersion:$sparkVersion")
                implementation(libs.s3mock.testcontainers)
                implementation(libs.testcontainers.juputer)
                runtimeOnly(libs.slf4j.simple)
                if (scalaVersion == "2.12") {
                    // Spark 3.5 marks javax.servlet-api as provided; needed at test runtime for MetricsServlet
                    runtimeOnly("javax.servlet:javax.servlet-api:4.0.1")
                }
            }
        }
    }
}

mavenPublishing {
    coordinates(groupId = "dev.vortex", artifactId = "vortex-spark_$scalaVersion", version = "${rootProject.version}")

    publishToMavenCentral()

    signAllPublications()

    pom {
        name = "vortex-spark_$scalaVersion"
        description = project.description
        url = "https://vortex.dev"
        inceptionYear = "2025"

        licenses {
            license {
                name = "Apache-2.0"
                url = "https://spdx.org/licenses/Apache-2.0.html"
            }
        }
        developers {
            developer {
                id = "spiraldb"
                name = "Vortex Authors"
            }
        }
        scm {
            connection = "scm:git:https://github.com/spiraldb/vortex.git"
            developerConnection = "scm:git:ssh://github.com/spiraldb/vortex.git"
            url = "https://github.com/spiraldb/vortex"
        }
    }

    repositories {
        mavenCentral()
        mavenLocal()
    }
}

// shade guava and protobuf dependencies
tasks.withType<ShadowJar> {
    relocate("com.google.protobuf", "dev.vortex.relocated.com.google.protobuf")
    relocate("com.google.common", "dev.vortex.relocated.com.google.common")
    relocate("org.apache.arrow", "dev.vortex.relocated.org.apache.arrow") {
        // exclude C Data Interface since JNI cannot be relocated
        exclude("org.apache.arrow.c.jni.JniWrapper")
        exclude("org.apache.arrow.c.jni.PrivateData")
        exclude("org.apache.arrow.c.jni.CDataJniException")
        // Also used by JNI: https://github.com/apache/arrow/blob/apache-arrow-11.0.0/java/c/src/main/cpp/jni_wrapper.cc#L341
        // Note this class is not used by us, but required when loading the native lib
        exclude("org.apache.arrow.c.ArrayStreamExporter\$ExportedArrayStreamPrivateData")
    }
    relocate("com.fasterxml.jackson", "dev.vortex.relocated.com.fasterxml.jackson")
}

tasks.withType<Test>().all {
    classpath +=
        project(":vortex-jni")
            .tasks
            .named("shadowJar")
            .get()
            .outputs.files
    jvmArgs(
        "--add-opens=java.base/sun.nio.ch=ALL-UNNAMED",
        "--add-opens=java.base/java.nio=ALL-UNNAMED",
        "--add-opens=java.base/sun.util.calendar=ALL-UNNAMED",
        "--add-opens=java.base/sun.security.action=ALL-UNNAMED",
    )
}

tasks.build {
    dependsOn("shadowJar")
}

description = "Apache Spark bindings for reading Vortex file datasets"
