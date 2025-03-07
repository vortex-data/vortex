import net.ltgt.gradle.errorprone.errorprone

plugins {
    id("com.diffplug.spotless") version "7.0.2"
    id("com.palantir.consistent-versions") version "2.31.0"
    id("com.palantir.git-version") version "3.1.0"
    id("io.github.gradle-nexus.publish-plugin") version "1.3.0"
    id("net.ltgt.errorprone") version "4.1.0" apply false
    id("org.inferred.processors") version "3.7.0" apply false
}

val gitVersion: groovy.lang.Closure<String> by extra
version = gitVersion()

nexusPublishing {
    repositories {
        sonatype {
            nexusUrl.set(uri("https://s01.oss.sonatype.org/service/local/"))
            snapshotRepositoryUrl.set(uri("https://s01.oss.sonatype.org/content/repositories/snapshots/"))
            username.set(System.getenv("MAVEN_CENTRAL_USER"))
            password.set(System.getenv("MAVEN_CENTRAL_PASSWORD"))
        }
    }
}

allprojects {
    apply(plugin = "com.diffplug.spotless")

    group = "dev.vortex"
    version = rootProject.version

    repositories {
        mavenCentral()
    }

    tasks.withType<Test> {
        useJUnitPlatform()

        maxHeapSize = "1G"

        testLogging {
            events("passed")
        }
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
            }
        }

        tasks.withType<JavaCompile> {
            options.errorprone.disable("UnusedVariable")
            options.release = 11

            // Needed to make sure that the barista-annotations emits to the correct directory
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
