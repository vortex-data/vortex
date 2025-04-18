import net.ltgt.gradle.errorprone.errorprone

plugins {
    id("com.diffplug.spotless") version "7.0.3"
    id("com.palantir.consistent-versions") version "2.32.0"
    id("com.palantir.git-version") version "3.2.0"
    id("net.ltgt.errorprone") version "4.2.0" apply false
    id("org.inferred.processors") version "3.7.0" apply false
    id("com.google.protobuf") version "0.9.5" apply false
    id("com.vanniktech.maven.publish") version "0.31.0" apply false
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
        apply(plugin = "org.inferred.processors")

        dependencies {
            "errorprone"("com.google.errorprone:error_prone_core")
            "errorprone"("com.jakewharton.nopen:nopen-checker")
            "compileOnly"("com.jakewharton.nopen:nopen-annotations")
        }

        spotless {
            java {
                palantirJavaFormat()
                licenseHeaderFile("${rootProject.projectDir}/.spotless/java-license-header.txt")
                targetExclude("**/generated/**")
                targetExcludeIfContentContains("// spotless:disabled")
            }
        }

        tasks.withType<JavaCompile> {
            options.errorprone.disable("UnusedVariable")
            options.errorprone.disableWarningsInGeneratedCode = true
            // ignore protobuf generated files
            options.errorprone.excludedPaths = ".*/build/generated/.*"
            options.release = 11

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
