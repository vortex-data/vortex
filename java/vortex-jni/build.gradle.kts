// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

import com.github.jengelman.gradle.plugins.shadow.tasks.ShadowJar
import org.gradle.kotlin.dsl.support.serviceOf

plugins {
    `java-library`
    `jvm-test-suite`
    id("com.google.protobuf")
    id("com.gradleup.shadow") version "9.4.1"
}

dependencies {
    // Align Netty versions across Arrow and Spark
    implementation(platform(libs.netty.bom))

    implementation(libs.arrow.c.data)
    implementation(libs.arrow.memory.core)
    implementation(libs.arrow.memory.netty)

    compileOnly(libs.immutables.value)
    annotationProcessor(libs.immutables.value)

    errorprone(libs.errorprone.core)
    errorprone(libs.nopen.checker)

    implementation(libs.guava)
    implementation(libs.protobuf.java)
    compileOnly(libs.errorprone.annotations)
    compileOnly(libs.nopen.annotations)

    // Logging
    implementation(libs.slf4j.api)
    testRuntimeOnly(libs.logback.classic)
}

testing {
    suites {
        val test by getting(JvmTestSuite::class) {
            useJUnitJupiter()
            dependencies {
                implementation(libs.junit.jupiter.params)
            }
        }
    }
}

mavenPublishing {
    coordinates(groupId = "dev.vortex", artifactId = "vortex-jni", version = "${rootProject.version}")
    publishToMavenCentral()

    signAllPublications()

    pom {
        name = "vortex-jni"
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

tasks.withType<Test>().all {
    jvmArgs(
        "--add-opens=java.base/java.nio=org.apache.arrow.memory.core,ALL-UNNAMED",
        "--add-opens=java.base/sun.nio.ch=ALL-UNNAMED",
        "--add-opens=java.base/java.nio=ALL-UNNAMED",
    )
}

protobuf {
    protoc {
        artifact = "com.google.protobuf:protoc:${libs.versions.protobuf.get()}"
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

tasks.build {
    dependsOn("shadowJar")
}

tasks.register("makeTestFiles") {
    description = "Generate files used by unit tests"
    group = "verification"

    doLast {
        println("makeTestFiles executed")

        val execOps = serviceOf<ExecOperations>()

        // Build the JNI lib
        execOps.exec {
            workingDir = rootProject.projectDir.absoluteFile.parentFile
            executable = "cargo"
            args("build", "--package", "vortex-jni")
        }

        copy {
            from("${rootProject.projectDir.absoluteFile.parentFile}/target/debug/libvortex_jni.so")
            into("$projectDir/src/main/resources/native/linux-amd64")
        }

        copy {
            from("${rootProject.projectDir.absoluteFile.parentFile}/target/debug/libvortex_jni.so")
            into("$projectDir/src/main/resources/native/linux-aarch64")
        }

        copy {
            from("${rootProject.projectDir.absoluteFile.parentFile}/target/debug/libvortex_jni.dylib")
            into("$projectDir/src/main/resources/native/darwin-aarch64")
        }

        execOps.exec {
            workingDir = rootProject.projectDir.absoluteFile.parentFile
            executable = "cargo"
            args("xtask", "java-test-files")
        }
    }
}

tasks.named("processResources").configure {
    dependsOn("makeTestFiles")
}

tasks.withType<Test>().all {
    dependsOn("makeTestFiles")
}

tasks.register("generateJniHeaders") {
    description = "Generates JNI header files for Java classes with native methods"
    group = "build"

    // Define input and output properties
    val jniClasses =
        fileTree("src/main/java") {
            // Adjust this include pattern to match only files that need JNI headers
            include("**/JNI*.java")
        }

    inputs.files(jniClasses)
    outputs.dir("${layout.buildDirectory}/generated/jni")

    doLast {
        // Create output directory if it doesn't exist
        val headerDir = file("${layout.buildDirectory}/generated/jni")
        headerDir.mkdirs()

        val classesDir =
            sourceSets["main"]
                .java.destinationDirectory
                .get()
                .asFile

        // Compile only the selected files with -h option
        ant.withGroovyBuilder {
            "javac"(
                "classpath" to sourceSets["main"].compileClasspath.asPath,
                "srcdir" to "src/main/java",
                "includes" to jniClasses.includes.joinToString(","),
                "destdir" to classesDir,
                "includeantruntime" to false,
                "debug" to true,
                "source" to java.sourceCompatibility,
                "target" to java.targetCompatibility,
            ) {
                "compilerarg"("line" to "-h ${headerDir.absolutePath}")
            }
        }

        println("JNI headers generated in ${headerDir.absolutePath}")
    }

    // Make this task run after the compileJava task
    dependsOn("compileJava")
}

description = "JNI bindings for the Vortex format"
