plugins {
    `java-library`
}

dependencies {
    api("org.apache.spark:spark-catalyst_2.12")
    api("org.apache.spark:spark-sql_2.12")
    api(project(":vortex-jni"))

    implementation("com.google.guava:guava")
    testImplementation("org.junit.jupiter:junit-jupiter")
}

tasks.withType<Test>().all {
    jvmArgs(
        "--add-opens=java.base/sun.nio.ch=ALL-UNNAMED",
        "--add-opens=java.base/java.nio=ALL-UNNAMED",
    )
}
