// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

import com.github.jengelman.gradle.plugins.shadow.tasks.ShadowJar
import net.ltgt.gradle.errorprone.errorprone
import org.gradle.kotlin.dsl.support.serviceOf

plugins {
    `java-library`
    `jvm-test-suite`
    id("com.gradleup.shadow") version "9.4.2"
    id("me.champeau.jmh") version "0.7.3"
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
    compileOnly(libs.errorprone.annotations)
    compileOnly(libs.nopen.annotations)
    api(libs.roaringbitmap)

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

    if (!project.hasProperty("skip.signing")) {
        signAllPublications()
    }

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

// shade guava and arrow dependencies in the published jar only. The JMH benchmark links the real
// (unrelocated) Arrow classes, so its jar must not be relocated — scope this to the `shadowJar` task.
tasks.named<ShadowJar>("shadowJar") {
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

    // The publish workflow places release, cross-compiled libs for every supported
    // architecture before invoking shadowJar; rebuilding the host-arch debug lib
    // here would overwrite them (linux-aarch64 ends up holding a linux-amd64 .so).
    onlyIf { System.getenv("VORTEX_SKIP_MAKE_TEST_FILES") != "true" }

    doLast {
        println("makeTestFiles executed")

        val execOps = serviceOf<ExecOperations>()

        // Build the JNI lib for the host architecture only.
        execOps.exec {
            workingDir = rootProject.projectDir.absoluteFile.parentFile
            executable = "cargo"
            args("build", "--package", "vortex-jni")
        }

        val osName = System.getProperty("os.name").lowercase()
        val osArch = System.getProperty("os.arch").lowercase()
        val osShortName =
            when {
                osName.contains("mac") -> "darwin"
                osName.contains("nix") || osName.contains("nux") -> "linux"
                osName.contains("win") -> "win"
                else -> throw GradleException("Unsupported OS for makeTestFiles: $osName")
            }
        val libExt =
            when (osShortName) {
                "darwin" -> ".dylib"
                "linux" -> ".so"
                "win" -> ".dll"
                else -> throw GradleException("Unsupported OS short name: $osShortName")
            }

        // Only populate the host-arch directory so cross-compiled libs for other
        // architectures (placed by the publish workflow) are preserved.
        copy {
            from("${rootProject.projectDir.absoluteFile.parentFile}/target/debug/libvortex_jni$libExt")
            into("$projectDir/src/main/resources/native/$osShortName-$osArch")
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

// ---------------------------------------------------------------------------
// JMH benchmarks (src/jmh). See BENCHMARKS.md.
//
// The read-boundary benchmark is meaningless against a debug native lib, so the `jmh` task builds
// and stages the release_debug cdylib itself (buildJmhNativeLib) rather than reusing the dev
// `makeTestFiles` debug build. The benchmark links the real Arrow classes off the runtime classpath
// (it is not run from the relocated shadowJar), so no relocation applies to it.
// ---------------------------------------------------------------------------
// Shared canonical benchmark file, generated by the Rust side and read by BOTH the JMH benchmark and
// the Rust `read_boundary` Divan bench so the two measure reads of the exact same bytes.
val workspaceRoot = rootProject.projectDir.absoluteFile.parentFile
val benchFile = workspaceRoot.resolve("target/vortex-jni-bench/data.vortex")

jmh {
    jmhVersion.set("1.37")
    // These reach the forked benchmark JVM. The Arrow C Data Interface needs the --add-opens; the
    // system property points the benchmark at the shared canonical file.
    jvmArgsAppend.addAll(
        "--add-opens=java.base/java.nio=ALL-UNNAMED",
        "--add-opens=java.base/sun.nio.ch=ALL-UNNAMED",
        "-Dvortex.jni.bench.file=${benchFile.absolutePath}",
    )
}

// Generate the shared canonical .vortex file via the Rust generator example. Idempotent: skipped
// while the file exists (delete it to regenerate). Both `jmh` and the Rust bench read this file.
val generateBenchFile =
    tasks.register("generateBenchFile") {
        description = "Generate the shared canonical .vortex file read by the JMH and Rust read benchmarks"
        group = "verification"

        outputs.file(benchFile)

        doLast {
            benchFile.parentFile.mkdirs()
            serviceOf<ExecOperations>().exec {
                workingDir = workspaceRoot
                executable = "cargo"
                args(
                    "run",
                    "--profile",
                    "release_debug",
                    "--quiet",
                    "--package",
                    "vortex-jni",
                    "--example",
                    "gen_bench_data",
                    "--",
                    benchFile.absolutePath,
                )
            }
        }
    }

// JMH benchmark classes/methods must be public and non-final, which the nopen checker forbids, and
// the generated JMH glue trips error-prone under -Werror. Relax both for the jmh source set only;
// main and test keep full strictness.
tasks.withType<JavaCompile>().configureEach {
    if (name.lowercase().contains("jmh")) {
        options.errorprone.enabled.set(false)
        options.compilerArgs.remove("-Werror")
    }
}

// Skip the redundant debug `makeTestFiles` build when this invocation runs the benchmark; the
// benchmark consumes the release_debug lib staged by buildJmhNativeLib instead.
val benchmarkRequested = objects.property(Boolean::class.java).convention(false)
gradle.taskGraph.whenReady {
    benchmarkRequested.set(allTasks.any { it.project == project && (it.name == "jmh" || it.name == "jmhJar") })
}
tasks.named("makeTestFiles").configure {
    onlyIf { !benchmarkRequested.get() }
}

val buildJmhNativeLib =
    tasks.register("buildJmhNativeLib") {
        description = "Build the release_debug vortex-jni cdylib and stage it for the JMH benchmark"
        group = "verification"

        // Stage on top of the processed resources so the benchmark loads it from the runtime classpath.
        dependsOn("processResources")

        doLast {
            val workspaceRoot = rootProject.projectDir.absoluteFile.parentFile

            serviceOf<ExecOperations>().exec {
                workingDir = workspaceRoot
                executable = "cargo"
                args("build", "--profile", "release_debug", "--package", "vortex-jni")
            }

            val osName = System.getProperty("os.name").lowercase()
            val osArch = System.getProperty("os.arch").lowercase()
            val osShortName =
                when {
                    osName.contains("mac") -> "darwin"
                    osName.contains("nix") || osName.contains("nux") -> "linux"
                    osName.contains("win") -> "win"
                    else -> throw GradleException("Unsupported OS for buildJmhNativeLib: $osName")
                }
            val libExt =
                when (osShortName) {
                    "darwin" -> ".dylib"
                    "linux" -> ".so"
                    "win" -> ".dll"
                    else -> throw GradleException("Unsupported OS short name: $osShortName")
                }

            copy {
                from("$workspaceRoot/target/release_debug/libvortex_jni$libExt")
                into(layout.buildDirectory.dir("resources/main/native/$osShortName-$osArch"))
            }
        }
    }

tasks.named("jmh").configure {
    dependsOn(buildJmhNativeLib)
    dependsOn(generateBenchFile)
}
tasks.named("jmhJar").configure { dependsOn(buildJmhNativeLib) }

// Standalone read-batch-granularity diagnostic (VortexJniBatchDiagnostic, not a JMH benchmark). Run
// it off the jmh runtime classpath, which carries the real Arrow classes and the staged
// release_debug lib (me.champeau.jmh's fat `jmhJar` does not bundle deps under com.gradleup.shadow).
tasks.register<JavaExec>("batchDiagnostic") {
    description = "Run the standalone read-batch-granularity diagnostic (VortexJniBatchDiagnostic)"
    group = "verification"
    dependsOn("buildJmhNativeLib")
    classpath = sourceSets["jmh"].runtimeClasspath
    mainClass.set("dev.vortex.bench.VortexJniBatchDiagnostic")
    jvmArgs(
        "--add-opens=java.base/java.nio=ALL-UNNAMED",
        "--add-opens=java.base/sun.nio.ch=ALL-UNNAMED",
    )
}

description = "JNI bindings for the Vortex format"
