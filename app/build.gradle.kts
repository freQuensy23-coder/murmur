plugins {
    alias(libs.plugins.android.application)
    // No org.jetbrains.kotlin.android: AGP 9 has built-in Kotlin.
    alias(libs.plugins.kotlin.compose)
    alias(libs.plugins.kotlin.serialization)
}

val productName = providers.gradleProperty("murmur.productName").getOrElse("Murmur")
// Device builds (DEVICE_CHECKLIST DC8): -Pmurmur.htp=true -Pmurmur.qnn=true. Both flags gate libraries whose
// licences allow object-code distribution only inside the application (ADR-048); the APK is the only shape they
// may leave in - never a standalone library download.
val shipHtp = providers.gradleProperty("murmur.htp").map { it.toBoolean() }.getOrElse(false)

android {
    namespace = "app.murmur"

    compileSdk {
        version = release(libs.versions.compileSdk.get().toInt()) {
            minorApiLevel = libs.versions.compileSdkMinor.get().toInt()
        }
    }
    buildToolsVersion = libs.versions.buildTools.get()
    ndkVersion = libs.versions.ndk.get()

    defaultConfig {
        applicationId = "app.murmur"
        minSdk = libs.versions.minSdk.get().toInt()
        targetSdk = libs.versions.targetSdk.get().toInt()
        versionCode = 1
        versionName = "0.1.0"
        // MurmurTestRunner = AndroidJUnitRunner + "-e murmurFake seeded|fresh|off" (debug fake container, WP-15 shell tests).
        testInstrumentationRunner = "app.murmur.shell.MurmurTestRunner"
        // D10: the product name lives in gradle.properties only.
        resValue("string", "app_name", productName)
        manifestPlaceholders["productName"] = productName
    }

    buildTypes {
        debug {
            // D6: the emulator runs the complete pipeline on CPU engines (x86_64), the phone on arm64.
            ndk { abiFilters += listOf("arm64-v8a", "x86_64") }
        }
        release {
            ndk { abiFilters += listOf("arm64-v8a") }
            isMinifyEnabled = false
            proguardFiles(getDefaultProguardFile("proguard-android-optimize.txt"))
        }
    }

    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_17
        targetCompatibility = JavaVersion.VERSION_17
    }

    buildFeatures {
        compose = true
        buildConfig = true
        resValues = true
    }

    packaging {
        jniLibs {
            // ADR-007: native libs must exist as files on disk. ggml-hexagon resolves its DSP skel
            // (libggml-htp-v79.so) and the CPU-variant loader scans a directory; QNN's skel needs
            // ADSP_LIBRARY_PATH to point at a real directory (llm spike 3.4, asr-ocr 5.3).
            useLegacyPackaging = true
            // ADR-048: the Hexagon-SDK-built DSP skel ships only with -Pmurmur.htp=true (licence unread, needsFromOwner),
            // the same rule as QNN (-Pmurmur.qnn=true, WP-06). Without it the :llm probe reports no HTP0 (CPU/GPU only).
            if (!shipHtp) excludes += "**/libggml-htp-*.so"
        }
    }

    providers.gradleProperty("murmur.streamAssets").orNull?.let {
        sourceSets.getByName("androidTest").assets.srcDir(file(it))
    }

    testOptions {
        unitTests.isReturnDefaultValues = true
    }
}

dependencies {
    // Every module is wired here and only here (AppContainer); features never depend on each other.
    implementation(project(":core:model"))
    implementation(project(":core:catalog"))
    implementation(project(":core:context"))
    implementation(project(":core:prompt"))
    implementation(project(":core:ocr"))
    implementation(project(":core:audio"))
    implementation(project(":core:engine"))
    implementation(project(":core:store"))
    implementation(project(":core:benchmark"))
    implementation(project(":core:dictation"))
    implementation(project(":infer:api"))
    implementation(project(":infer:asr"))
    implementation(project(":infer:llm"))
    implementation(project(":data:db"))
    implementation(project(":data:models"))
    implementation(project(":platform:a11y"))
    implementation(project(":platform:overlay"))
    implementation(project(":platform:audio"))
    implementation(project(":ui:design"))
    implementation(project(":feature:history"))
    implementation(project(":feature:dashboard"))
    implementation(project(":feature:models"))
    implementation(project(":feature:settings"))
    implementation(project(":feature:onboarding"))
    // D6: debug builds can run the whole pipeline on deterministic fakes, and the debug fake container swaps the data
    // side for screenshot and shell tests (DebugWiring in src/debug). :testing never reaches the release APK.
    debugImplementation(project(":testing"))

    implementation(libs.androidx.core.ktx)
    implementation(libs.androidx.lifecycle.runtime.ktx)
    implementation(libs.androidx.lifecycle.viewmodel.compose)
    implementation(libs.androidx.activity.compose)
    implementation(libs.kotlinx.coroutines.android)
    implementation(libs.kotlinx.serialization.json)
    implementation(libs.androidx.navigation3.runtime)
    implementation(libs.androidx.navigation3.ui)
    // ViewModels scoped to back-stack entries (NavDisplay's entry decorator).
    implementation(libs.androidx.lifecycle.viewmodel.navigation3)
    implementation(libs.androidx.material3.adaptive)
    implementation(libs.androidx.material3.adaptive.layout)
    implementation(libs.androidx.material3.adaptive.navigation3)

    implementation(platform(libs.androidx.compose.bom))
    implementation(libs.androidx.compose.ui)
    implementation(libs.androidx.compose.ui.graphics)
    implementation(libs.androidx.compose.ui.tooling.preview)
    implementation(libs.androidx.compose.material3)
    debugImplementation(libs.androidx.compose.ui.tooling)
    debugImplementation(libs.androidx.compose.ui.test.manifest)

    testImplementation(libs.junit)
    testImplementation(project(":testing"))
    testImplementation(libs.kotlinx.coroutines.test)

    androidTestImplementation(platform(libs.androidx.compose.bom))
    androidTestImplementation(libs.androidx.compose.ui.test.junit4)
    androidTestImplementation(libs.androidx.test.ext.junit)
    androidTestImplementation(libs.androidx.test.runner)
    androidTestImplementation(libs.androidx.test.uiautomator)
    androidTestImplementation(project(":testing"))
}
