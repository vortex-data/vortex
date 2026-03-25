// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

import com.github.jengelman.gradle.plugins.shadow.tasks.ShadowJar

apply(plugin = "com.vanniktech.maven.publish")

plugins {
    `java-library`
    `jvm-test-suite`
    id("com.gradleup.shadow") version "9.4.0"
}

dependencies {
    compileOnly("org.apache.spark:spark-catalyst_2.13")
    compileOnly("org.apache.spark:spark-sql_2.13")
    api(project(":vortex-jni", configuration = "shadow"))

    compileOnly("org.immutables:value")
    annotationProcessor("org.immutables:value")

    implementation("com.google.guava:guava")
    implementation("org.slf4j:slf4j-api:2.0.17")
}

testing {
    suites {
        val test by getting(JvmTestSuite::class) {
            useJUnitJupiter()

            dependencies {
                implementation("org.junit.jupiter:junit-jupiter:6.0.3")
                implementation("org.apache.spark:spark-core_2.13")
                implementation("org.apache.spark:spark-sql_2.13")
                runtimeOnly("org.slf4j:slf4j-simple:2.0.17")
                // S3Mock Testcontainers for testing S3 integration (avoids classpath conflicts)
                implementation("com.adobe.testing:s3mock-testcontainers")
                implementation("org.testcontainers:junit-jupiter")
            }
        }
    }
}

mavenPublishing {
    coordinates(groupId = "dev.vortex", artifactId = "vortex-spark", version = "${rootProject.version}")

    publishToMavenCentral()

    signAllPublications()

    pom {
        name = "vortex-spark"
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
