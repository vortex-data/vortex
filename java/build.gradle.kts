// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

import net.ltgt.gradle.errorprone.errorprone

plugins {
    id("com.diffplug.spotless") version "8.6.0"
    id("com.palantir.git-version") version "5.0.0"
    id("com.palantir.java-format") version "2.93.0"
    id("net.ltgt.errorprone") version "5.1.0" apply false
}

spotless {
    java {
        target(fileTree("vortex-spark/common") { include("**/*.java") })
        palantirJavaFormat().formatJavadoc(true)
        licenseHeaderFile("${rootProject.projectDir}/.spotless/java-license-header.txt")
        removeUnusedImports()
        forbidWildcardImports()
        importOrder("")
        trimTrailingWhitespace()
        leadingTabsToSpaces(4)
        targetExclude("**/generated/**")
        targetExcludeIfContentContains("// spotless:disabled")
    }
    scala {
        target(fileTree("vortex-spark/common") { include("**/*.scala") })
        scalafmt("3.9.10")
        licenseHeaderFile(
            "${rootProject.projectDir}/.spotless/java-license-header.txt",
            "package ",
        )
    }
}

subprojects {
    apply(plugin = "com.vanniktech.maven.publish")
}

val gitVersion: groovy.lang.Closure<String> by extra
version = gitVersion()

allprojects {
    apply(plugin = "com.diffplug.spotless")

    group = "dev.vortex"
    version = rootProject.version

    repositories {
        mavenCentral()
    }

    plugins.withType<JavaLibraryPlugin> {
        apply(plugin = "net.ltgt.errorprone")

        dependencies {
            "errorprone"("com.google.errorprone:error_prone_core:2.36.0")
            "errorprone"("com.jakewharton.nopen:nopen-checker:1.0.1")
            "compileOnly"("com.jakewharton.nopen:nopen-annotations:1.0.1")
        }

        spotless {
            java {
                if (project.name.startsWith("vortex-spark-")) {
                    // Shared sources are formatted by the root project, where they are inside the project directory.
                    target(project.fileTree("src") { include("**/*.java") })
                }
                palantirJavaFormat().formatJavadoc(true)
                licenseHeaderFile("${rootProject.projectDir}/.spotless/java-license-header.txt")
                removeUnusedImports()
                forbidWildcardImports()
                importOrder("")
                trimTrailingWhitespace()
                leadingTabsToSpaces(4)
                targetExclude("**/generated/**")
                targetExcludeIfContentContains("// spotless:disabled")
            }
        }

        tasks.withType<JavaCompile> {
            options.errorprone.disable("UnusedVariable")
            options.errorprone.disableWarningsInGeneratedCode = true
            // JMH generates non-final subclasses of every benchmark, which ErrorProne's Nopen check rejects.
            options.errorprone.enabled.set(name != "compileBenchmarkJava")
            options.release = 17
            options.compilerArgs.add("-Werror")

            options.generatedSourceOutputDirectory = projectDir.resolve("generated_src")
        }

        tasks.withType<Javadoc> {
            (options as StandardJavadocDocletOptions).addBooleanOption("Xdoclint:-missing", true)
        }

        the<JavaPluginExtension>().toolchain {
            languageVersion.set(JavaLanguageVersion.of(17))
            vendor.set(JvmVendorSpec.AMAZON)
        }

        tasks["check"].dependsOn("spotlessCheck")
        if (project.name == "vortex-spark-4.0_2.13") {
            tasks["check"].dependsOn(rootProject.tasks.named("spotlessCheck"))
        }
    }

    spotless {
        kotlinGradle {
            ktlint()
        }
    }

    if (project.name.startsWith("vortex-spark-") && project.name != "vortex-spark-4.0_2.13") {
        // Spark variants share sources. Format them from the root project only.
        tasks.register("format") { enabled = false }
    } else {
        tasks.register("format").get().dependsOn(
            if (project.name == "vortex-spark-4.0_2.13") {
                rootProject.tasks.named("spotlessApply")
            } else {
                tasks.named("spotlessApply")
            },
        )
    }
}
