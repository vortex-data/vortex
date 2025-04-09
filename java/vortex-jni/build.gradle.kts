import com.github.jengelman.gradle.plugins.shadow.tasks.ShadowJar

plugins {
    `java-library`
    `jvm-test-suite`
    `maven-publish`
    id("com.google.protobuf")
    id("com.gradleup.shadow") version "8.3.6"
    id("me.champeau.jmh") version "0.7.3"
}

dependencies {
    implementation("org.apache.arrow:arrow-c-data")
    implementation("org.apache.arrow:arrow-memory-core")
    implementation("org.apache.arrow:arrow-memory-netty")

    compileOnly("org.immutables:value")
    annotationProcessor("org.immutables:value")

    errorprone("com.google.errorprone:error_prone_core")
    errorprone("com.jakewharton.nopen:nopen-checker")

    implementation("com.google.guava:guava")
    implementation("com.google.protobuf:protobuf-java")
    compileOnly("com.google.errorprone:error_prone_annotations")
    compileOnly("com.jakewharton.nopen:nopen-annotations")
}

jmh {
    warmupIterations = 3
    iterations = 3
    fork = 1
}

testing {
    suites {
        val test by getting(JvmTestSuite::class) {
            useJUnitJupiter()
        }
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
        artifact = "com.google.protobuf:protoc:4.30.2"
    }
}

// shade guava and protobuf dependencies
tasks.withType<ShadowJar> {
    archiveClassifier.set("")
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

tasks.build {
    dependsOn("shadowJar")
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

publishing {
    publications {
        create<MavenPublication>("mavenJava") {
            artifact(tasks.shadowJar.get())
            artifactId = "vortex-jni"
        }
    }
}

val vortexJNI = projectDir.parentFile.parentFile.resolve("vortex-jni")

val targetDir = projectDir.parentFile.parentFile.resolve("target")

// These are the native platforms that we want to build for and ship inside of our JAR
val rustTargets = listOf(
    "aarch64-apple-darwin",
    "x86_64-unknown-linux-gnu",
)

rustTargets.forEach { target ->
    tasks.register("cargoBuild_$target", Exec::class) {
        workingDir = vortexJNI

        commandLine(
            "cargo",
            "zigbuild",
            "--release",
            "--target",
            target,
            "--package",
            "vortex-jni",
        )

        val platformSuffix = if (target.contains("darwin")) {
            "dylib"
        } else {
            "so"
        }

        outputs.files(targetDir.resolve("$target/release/libvortex_jni.$platformSuffix"))

        // Always force rebuilds, rely on cargo's builtin caching and incremental compile to avoid spurious rebuilds.
        outputs.upToDateWhen { false }
        outputs.cacheIf { false }
    }
}

tasks.named("build").configure {
    rustTargets.map {
        val task = tasks.named("cargoBuild_$it")
        dependsOn(task)
    }
}

val copySharedLibrary by tasks.register("copySharedLibrary") {
    rustTargets.forEach { target ->
        val platformTask = tasks.named("cargoBuild_$target")
        dependsOn(platformTask)

        doLast {
            copy {
                println("copy task for $target executing")
                val arch = when (target.split("-")[0]) {
                    "amd64", "x86_64" -> "amd64"
                    else -> target.split("-")[0]
                }
                val resourceDir = if (target.contains("darwin")) {
                    "darwin-$arch"
                } else {
                    "linux-$arch"
                }
                from(platformTask.get().outputs.files)
                into(projectDir.resolve("src/main/resources/native/$resourceDir"))
            }
        }
    }
}

tasks.withType<ProcessResources>().configureEach {
    dependsOn(copySharedLibrary)
}

// Remove the JAR task, replace it with shadowJar
tasks.named("jar").configure {
    dependsOn("shadowJar")
    enabled = false
}
