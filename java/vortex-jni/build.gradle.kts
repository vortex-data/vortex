plugins {
    `java-library`
    `jvm-test-suite`
    `maven-publish`
    id("com.google.protobuf")
}

dependencies {
    api("net.java.dev.jna:jna-platform")
    api("com.google.protobuf:protobuf-java")

    compileOnly("org.immutables:value")
    annotationProcessor("org.immutables:value")

    errorprone("com.google.errorprone:error_prone_core")
    errorprone("com.jakewharton.nopen:nopen-checker")

    implementation("com.google.guava:guava")
    compileOnly("com.google.errorprone:error_prone_annotations")
    compileOnly("com.jakewharton.nopen:nopen-annotations")
}

testing {
    suites {
        val test by getting(JvmTestSuite::class) {
            useJUnitJupiter()
        }
    }
}

protobuf {
    protoc {
        artifact = "com.google.protobuf:protoc:4.30.1"
    }
}

publishing {
    publications {
        create<MavenPublication>("mavenJava") {
            from(components["java"]) // Publishes the compiled JAR
            artifactId = "vortex-jni"
        }
    }
}

val vortexFFI = projectDir.parentFile.parentFile.resolve("vortex-ffi")

val platformLibSuffix =
    if (System.getProperty("os.name").contains("Mac")) {
        "dylib"
    } else {
        "so"
    }

val targetDir = projectDir.parentFile.parentFile.resolve("target")
val libraryFile = targetDir.resolve("release/libvortex_ffi.$platformLibSuffix")

val cargoCheck by tasks.registering(Exec::class) {
    workingDir = vortexFFI
    commandLine("cargo", "check")
}

val cargoBuild by tasks.registering(Exec::class) {
    workingDir = vortexFFI
    commandLine(
        "cargo",
        "build",
        "--release",
    )

    outputs.files(libraryFile)

    // Always force rebuilds, rely on cargo's builtin caching and incremental compile to avoid spurious rebuilds.
    outputs.upToDateWhen { false }
    outputs.cacheIf { false }
}

val cargoClean by tasks.registering(Exec::class) {
    workingDir = vortexFFI
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
        "amd64", "x86_64" -> "x86-64"
        else -> System.getProperty("os.arch")
    }
val resourceDir =
    if (osName.startsWith("Mac")) {
        "darwin-$osArch"
    } else {
        "linux-$osArch"
    }

// Create a release build for every platform we care about.
// Or we distribute different JARs for each platform...not sure the best approach here.
// Honestly, fat JAR is probably the move. No one cares about JAR size in Java land, portability
// is more important.
val copySharedLibrary by tasks.registering(Copy::class) {
    dependsOn(cargoBuild)

    from(libraryFile)
    into(projectDir.resolve("src/main/resources/$resourceDir"))

    doLast {
        println("Copied $libraryFile into resource directory")
    }
}

tasks.withType<ProcessResources>().configureEach {
    dependsOn(copySharedLibrary)
}
