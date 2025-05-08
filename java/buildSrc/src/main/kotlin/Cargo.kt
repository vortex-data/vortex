import com.moandjiezana.toml.Toml
import org.gradle.api.Project

fun Project.cargoVersion(): String {
    val manifestFile = rootDir.parentFile.resolve("Cargo.toml")
    val manifest = Toml().read(manifestFile)

    return manifest.getString("workspace.package.version")
}
