import com.vanniktech.maven.publish.SonatypeHost

apply(plugin = "com.vanniktech.maven.publish")

plugins {
    `java-library`
    `jvm-test-suite`
}

dependencies {
    api("org.apache.spark:spark-catalyst_2.12")
    api("org.apache.spark:spark-sql_2.12")
    api(project(":vortex-jni", configuration = "shadow"))

    compileOnly("org.immutables:value")
    annotationProcessor("org.immutables:value")

    implementation("com.google.guava:guava")
}

testing {
    suites {
        val test by getting(JvmTestSuite::class) {
            useJUnitJupiter()
        }
    }
}

mavenPublishing {
    coordinates(groupId = "dev.vortex", artifactId = "vortex-spark", version = "${rootProject.version}")

    publishToMavenCentral(SonatypeHost.CENTRAL_PORTAL)

    signAllPublications()

    pom {
        name = "vortex-spark"
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
    classpath +=
        project(":vortex-jni")
            .tasks
            .named("shadowJar")
            .get()
            .outputs.files
    jvmArgs(
        "--add-opens=java.base/sun.nio.ch=ALL-UNNAMED",
        "--add-opens=java.base/java.nio=ALL-UNNAMED",
        "--add-opens=java.base/sun.util.calendar=ALL-UNNAMED",
    )
}

description = "Apache Spark bindings for reading Vortex file datasets"
