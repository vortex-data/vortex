// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

import net.ltgt.gradle.errorprone.errorprone

plugins {
    id("com.diffplug.spotless") version "8.5.1"
    id("com.palantir.git-version") version "5.0.0"
    id("com.palantir.java-format") version "2.90.0"
    id("net.ltgt.errorprone") version "5.1.0" apply false
    id("com.vanniktech.maven.publish") version "0.36.0" apply false
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
            options.release = 17
            options.compilerArgs.add("-Werror")

            options.generatedSourceOutputDirectory = projectDir.resolve("generated_src")
        }

        tasks.withType<Javadoc> {
            (options as StandardJavadocDocletOptions).addStringOption("Xdoclint:-missing")
        }

        the<JavaPluginExtension>().toolchain {
            languageVersion.set(JavaLanguageVersion.of(17))
            vendor.set(JvmVendorSpec.AMAZON)
        }

        tasks["check"].dependsOn("spotlessCheck")
    }

    spotless {
        kotlinGradle {
            ktlint()
        }
    }

    tasks.register("format").get().dependsOn("spotlessApply")
}
