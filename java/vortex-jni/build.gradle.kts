import com.github.jengelman.gradle.plugins.shadow.tasks.ShadowJar
import java.io.ByteArrayOutputStream

plugins {
    `java-library`
    `jvm-test-suite`
    `maven-publish`
    id("com.google.protobuf")
    id("com.gradleup.shadow") version "8.3.6"
    id("me.champeau.jmh") version "0.7.2"
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

val platformLibSuffix =
    if (System.getProperty("os.name").contains("Mac")) {
        "dylib"
    } else {
        "so"
    }

val targetDir = projectDir.parentFile.parentFile.resolve("target")

val cargoCheck by tasks.registering(Exec::class) {
    workingDir = vortexJNI
    commandLine("cargo", "check")
}

val cargoBuild by tasks.registering(Exec::class) {
    workingDir = vortexJNI

    val buildWithAsan = project.findProperty("buildWithAsan")?.toString()?.toBoolean() ?: false
    println("buildWithAsan: $buildWithAsan")
    if (buildWithAsan) {
        // Force a rebuild
        outputs.upToDateWhen { false }

        // Get the target triple for the current platform. We need it
        // so we can tell cargo to recompile the std crate for this target with ASAN enabled.
        val output = ByteArrayOutputStream()
        exec {
            commandLine("rustc", "--print", "host-tuple")
            standardOutput = output
        }
        val targetTriple = output.toString().trim()

        // Build with ASAN to detect memory leaks and out of bounds accesses from Java.
        environment("RUSTFLAGS", "-Zsanitizer=address")
        commandLine(
            "cargo",
            "build",
            "-Zbuild-std",
            "--target",
            targetTriple,
            "--release",
        )
        // cargo puts the built artifact in a target-dependent directory when you specify the triple
        outputs.files(targetDir.resolve("$targetTriple/release/libvortex_jni.$platformLibSuffix"))
    } else {
        commandLine(
            "cargo",
            "build",
            "--release",
        )
        outputs.files(targetDir.resolve("release/libvortex_jni.$platformLibSuffix"))
    }

    // Always force rebuilds, rely on cargo's builtin caching and incremental compile to avoid spurious rebuilds.
    outputs.upToDateWhen { false }
    outputs.cacheIf { false }
}

val cargoClean by tasks.registering(Exec::class) {
    workingDir = vortexJNI
    commandLine("cargo", "clean")
}

tasks.named("check").configure {
    dependsOn.add(cargoCheck)
}

tasks.named("build").configure {
    dependsOn.add(cargoBuild)
}

val osName = System.getProperty("os.name")
val osArch =
    when (System.getProperty("os.arch")) {
        "amd64", "x86_64" -> "amd64"
        else -> System.getProperty("os.arch")
    }
val resourceDir =
    if (osName.startsWith("Mac")) {
        "darwin-$osArch"
    } else {
        "linux-$osArch"
    }

val copySharedLibrary by tasks.registering(Copy::class) {
    dependsOn(cargoBuild)

    from(cargoBuild.get().outputs.files)
    into(projectDir.resolve("src/main/resources/native/$resourceDir"))
}

tasks.withType<ProcessResources>().configureEach {
    dependsOn(copySharedLibrary)
}

// Remove the JAR task, replace it with shadowJar
tasks.named("jar").configure {
    dependsOn("shadowJar")
    enabled = false
}
