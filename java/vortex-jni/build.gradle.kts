import com.github.jengelman.gradle.plugins.shadow.tasks.ShadowJar

plugins {
    `java-library`
    `jvm-test-suite`
    `maven-publish`
    id("com.google.protobuf")
    id("com.gradleup.shadow") version "8.3.6"
}

dependencies {
    compileOnly("org.immutables:value")
    annotationProcessor("org.immutables:value")

    errorprone("com.google.errorprone:error_prone_core")
    errorprone("com.jakewharton.nopen:nopen-checker")

    implementation("com.google.guava:guava")
    implementation("com.google.protobuf:protobuf-java")
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

// shade guava and protobuf dependencies
tasks.withType<ShadowJar> {
    archiveClassifier.set("")
    relocate("com.google.protobuf", "dev.vortex.relocated.com.google.protobuf")
    relocate("com.google.common", "dev.vortex.relocated.com.google.common")
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
    outputs.dir("$buildDir/generated/jni")

    doLast {
        // Create output directory if it doesn't exist
        val headerDir = file("$buildDir/generated/jni")
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
val libraryFile = targetDir.resolve("release/libvortex_jni.$platformLibSuffix")

val cargoCheck by tasks.registering(Exec::class) {
    workingDir = vortexJNI
    commandLine("cargo", "check")
}

val cargoBuild by tasks.registering(Exec::class) {
    workingDir = vortexJNI
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
    into(projectDir.resolve("src/main/resources/native/$resourceDir"))

    doLast {
        println("Copied $libraryFile into resource directory")
    }
}

tasks.withType<ProcessResources>().configureEach {
    dependsOn(copySharedLibrary)
}

// Remove the JAR task, replace it with shadowJar
tasks.named("jar").configure {
    enabled = false
}
