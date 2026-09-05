// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

import com.github.jengelman.gradle.plugins.shadow.tasks.ShadowJar
import org.gradle.api.tasks.javadoc.Javadoc
import org.gradle.api.tasks.scala.ScalaCompile

plugins {
    scala
    `java-library`
    `jvm-test-suite`
    id("com.gradleup.shadow")
    id("com.vanniktech.maven.publish")
}

val libs = extensions.getByType<VersionCatalogsExtension>().named("libs")
val sparkLine = project.name.removePrefix("vortex-spark-").substringBefore('_')
val scalaBinaryVersion = project.name.substringAfterLast('_')
val sparkVersion =
    when (sparkLine) {
        "3.5" -> libs.findVersion("spark35").get().requiredVersion
        "4.0" -> libs.findVersion("spark40").get().requiredVersion
        else -> throw GradleException("Unsupported Spark line: $sparkLine")
    }
val scalaVersion =
    when (sparkLine to scalaBinaryVersion) {
        "3.5" to "2.12" -> libs.findVersion("scala212").get().requiredVersion
        "3.5" to "2.13" -> libs.findVersion("scala213Spark35").get().requiredVersion
        "4.0" to "2.13" -> libs.findVersion("scala213Spark40").get().requiredVersion
        else -> throw GradleException("Unsupported Spark/Scala variant: $sparkLine/$scalaBinaryVersion")
    }
val spark41Version = libs.findVersion("spark41").get().requiredVersion

layout.buildDirectory = layout.projectDirectory.dir("build/${project.name}")

val commonSourceDir = rootProject.file("vortex-spark/common/src")
val versionSourceDir = project.file("src")

sourceSets {
    named("main") {
        // Java stays with javac so ErrorProne, Nopen, and -Werror keep covering it. Only Scala
        // depends on Java here, and compileScala already sees compileJava's output.
        java.setSrcDirs(
            listOf(
                commonSourceDir.resolve("main/java"),
                versionSourceDir.resolve("main/java"),
            ),
        )
        scala.setSrcDirs(
            listOf(
                commonSourceDir.resolve("main/scala"),
                versionSourceDir.resolve("main/scala"),
            ),
        )
        resources.setSrcDirs(
            listOf(
                commonSourceDir.resolve("main/resources"),
                versionSourceDir.resolve("main/resources"),
            ),
        )
    }
    named("test") {
        java.setSrcDirs(
            listOf(
                commonSourceDir.resolve("test/java"),
                versionSourceDir.resolve("test/java"),
            ),
        )
        scala.setSrcDirs(
            listOf(
                commonSourceDir.resolve("test/scala"),
                versionSourceDir.resolve("test/scala"),
            ),
        )
        resources.setSrcDirs(
            listOf(
                commonSourceDir.resolve("test/resources"),
                versionSourceDir.resolve("test/resources"),
            ),
        )
    }
}

dependencies {
    compileOnly("org.scala-lang:scala-library:$scalaVersion")
    compileOnly("org.apache.spark:spark-catalyst_$scalaBinaryVersion:$sparkVersion")
    compileOnly("org.apache.spark:spark-sql_$scalaBinaryVersion:$sparkVersion")
    api(project(":vortex-jni", configuration = "shadow"))

    implementation(libs.findLibrary("guava").get())
    implementation(libs.findLibrary("slf4j-api").get())
}

tasks.withType<ScalaCompile>().configureEach {
    scalaCompileOptions.additionalParameters = listOf("-release:17", "-deprecation", "-feature")
}

testing {
    suites {
        val test by getting(JvmTestSuite::class) {
            useJUnitJupiter()
            dependencies {
                implementation(libs.findLibrary("junit-jupiter").get())
                implementation("org.apache.spark:spark-core_$scalaBinaryVersion:$sparkVersion")
                implementation("org.apache.spark:spark-sql_$scalaBinaryVersion:$sparkVersion")
                implementation(libs.findLibrary("s3mock-testcontainers").get())
                implementation(libs.findLibrary("testcontainers-juputer").get())
                runtimeOnly(libs.findLibrary("slf4j-simple").get())
                if (sparkLine == "3.5") {
                    runtimeOnly("javax.servlet:javax.servlet-api:4.0.1")
                }
            }
        }

        // Benchmarks are compiled by javac so the JMH annotation processor can generate its
        // benchmark registry. Joint scalac compilation would skip that step.
        register<JvmTestSuite>("benchmark") {
            sources {
                java.setSrcDirs(
                    listOf(
                        commonSourceDir.resolve("jmh/java"),
                        versionSourceDir.resolve("jmh/java"),
                    ),
                )
                resources.setSrcDirs(emptyList<String>())
            }
            dependencies {
                implementation(project())
                implementation(libs.findLibrary("jmh-core").get())
                annotationProcessor(libs.findLibrary("jmh-generator-annprocess").get())
                implementation("org.apache.spark:spark-core_$scalaBinaryVersion:$sparkVersion")
                implementation("org.apache.spark:spark-sql_$scalaBinaryVersion:$sparkVersion")
                runtimeOnly(libs.findLibrary("slf4j-simple").get())
                if (sparkLine == "3.5") {
                    runtimeOnly("javax.servlet:javax.servlet-api:4.0.1")
                }
            }
            // The suite only carries JMH sources; its default test task has nothing to run.
            targets.all { testTask.configure { enabled = false } }
        }

        if (sparkLine == "4.0") {
            register<JvmTestSuite>("spark41CompatTest") {
                useJUnitJupiter()
                sources {
                    java.setSrcDirs(
                        listOf(
                            commonSourceDir.resolve("test/java"),
                            versionSourceDir.resolve("test/java"),
                        ),
                    )
                    resources.setSrcDirs(
                        listOf(
                            commonSourceDir.resolve("test/resources"),
                            versionSourceDir.resolve("test/resources"),
                        ),
                    )
                }
                dependencies {
                    implementation(project())
                    implementation(libs.findLibrary("junit-jupiter").get())
                    implementation("org.apache.spark:spark-core_2.13:$spark41Version")
                    implementation("org.apache.spark:spark-sql_2.13:$spark41Version")
                    implementation(libs.findLibrary("s3mock-testcontainers").get())
                    implementation(libs.findLibrary("testcontainers-juputer").get())
                    runtimeOnly(libs.findLibrary("slf4j-simple").get())
                }
            }
        }
    }
}

if (sparkLine == "4.0") {
    tasks.named("check") {
        dependsOn("spark41CompatTest")
    }
}

mavenPublishing {
    coordinates(
        groupId = "dev.vortex",
        artifactId = project.name,
        version = rootProject.version.toString(),
    )
    publishToMavenCentral()
    if (!project.hasProperty("skip.signing")) {
        signAllPublications()
    }
    repositories {
        mavenCentral()
        mavenLocal()
    }
    pom {
        name = project.name
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
}

tasks.withType<ShadowJar>().configureEach {
    relocate("com.google.common", "dev.vortex.relocated.com.google.common")
    relocate("org.apache.arrow", "dev.vortex.relocated.org.apache.arrow") {
        exclude("org.apache.arrow.c.jni.JniWrapper")
        exclude("org.apache.arrow.c.jni.PrivateData")
        exclude("org.apache.arrow.c.jni.CDataJniException")
        exclude("org.apache.arrow.c.ArrayStreamExporter\$ExportedArrayStreamPrivateData")
    }
    relocate("com.fasterxml.jackson", "dev.vortex.relocated.com.fasterxml.jackson")
}

val sparkJvmArgs =
    listOf(
        "--add-opens=java.base/sun.nio.ch=ALL-UNNAMED",
        "--add-opens=java.base/java.nio=ALL-UNNAMED",
        "--add-opens=java.base/sun.util.calendar=ALL-UNNAMED",
        "--add-opens=java.base/sun.security.action=ALL-UNNAMED",
    )

tasks.withType<Test>().configureEach {
    classpath += project(":vortex-jni").tasks.named("shadowJar").get().outputs.files
    jvmArgs(sparkJvmArgs)
}

tasks.register<JavaExec>("jmh") {
    description = "Run JMH benchmarks. Pass JMH arguments with -PjmhArgs=\"<regex> -f 1 -wi 2 -i 3\"."
    group = "verification"

    val benchmarkSourceSet = sourceSets.named("benchmark").get()
    dependsOn(benchmarkSourceSet.classesTaskName)
    classpath = benchmarkSourceSet.runtimeClasspath + project(":vortex-jni").tasks.named("shadowJar").get().outputs.files
    mainClass = "org.openjdk.jmh.Main"
    jvmArgs(sparkJvmArgs)
    val jmhArgs = project.findProperty("jmhArgs")?.toString() ?: ""
    args(jmhArgs.split(' ').filter { it.isNotBlank() })
}

tasks.withType<Javadoc>().configureEach {
    setSource(
        files(
            fileTree(commonSourceDir.resolve("main/java")) { include("**/*.java") },
            fileTree(versionSourceDir.resolve("main/java")) { include("**/*.java") },
        ),
    )
}

tasks.named("build") {
    dependsOn("shadowJar")
}

description = "Apache Spark $sparkLine bindings for reading and writing Vortex file datasets"
