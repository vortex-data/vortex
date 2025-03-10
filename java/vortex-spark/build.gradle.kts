plugins {
    `java-library`
    `maven-publish`
}

dependencies {
    api("org.apache.spark:spark-catalyst_2.12")
    api("org.apache.spark:spark-sql_2.12")
    api(project(":vortex-jni"))

    compileOnly("org.immutables:value")
    annotationProcessor("org.immutables:value")

    implementation("com.google.guava:guava")
    testImplementation("org.junit.jupiter:junit-jupiter")
}

publishing {
    publications {
        create<MavenPublication>("mavenJava") {
            from(components["java"]) // Publishes the compiled JAR
            artifactId = "vortex-spark"
        }
    }
}

tasks.withType<Test>().all {
    jvmArgs(
        "--add-opens=java.base/sun.nio.ch=ALL-UNNAMED",
        "--add-opens=java.base/java.nio=ALL-UNNAMED",
        "--add-opens=java.base/sun.util.calendar=ALL-UNNAMED",
    )
}
