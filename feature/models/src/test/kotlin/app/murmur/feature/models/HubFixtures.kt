package app.murmur.feature.models

import app.murmur.core.catalog.Catalog
import app.murmur.core.catalog.CatalogParser
import app.murmur.core.catalog.DeviceProfile
import app.murmur.core.catalog.ModelEntry
import app.murmur.core.catalog.UnitHealth
import app.murmur.core.model.ArtifactId
import app.murmur.core.model.ComputeUnit
import app.murmur.core.model.ModelKind
import app.murmur.core.store.BenchmarkRecord
import app.murmur.core.store.BenchmarkResultData
import app.murmur.core.store.InstallState
import app.murmur.core.store.Settings
import app.murmur.core.store.StorageInfo
import app.murmur.testing.FakeDeviceProfileRepository

/**
 * Test inputs built from the bundled catalog. Entries are found by facts (kind, units, traits), never by id, so the
 * tests keep working when the catalog changes and the module stays free of model literals.
 */
object HubFixtures {
    val catalog: Catalog = CatalogParser.bundled()
    val emulator: DeviceProfile = FakeDeviceProfileRepository.EMULATOR
    val fold7: DeviceProfile = FakeDeviceProfileRepository.FOLD7

    /** The catalog's default ru+en speech model (Parakeet today): an NPU artifact for SM8750 and a CPU artifact. */
    val npuAndCpuAsr: ModelEntry get() = catalog.model(catalog.defaults.asr.getValue("ru+en"))!!.also {
        require(it.artifacts.any { a -> ComputeUnit.NPU in a.units && a.socs.isNotEmpty() } && it.artifacts.any { a -> a.units == listOf(ComputeUnit.CPU) })
    }

    /** A streaming preview-only model (ADR-047). */
    val previewOnly: ModelEntry get() = catalog.models.first { it.asr?.previewOnly == true && it.artifacts.any { a -> a.units == listOf(ComputeUnit.CPU) } }

    val defaultLlm: ModelEntry get() = catalog.models.first { it.llm != null && it.artifacts.any { a -> ComputeUnit.NPU in a.units } }

    /** Another available LLM with a Russian-capable card (catalog order). */
    val otherLlm: ModelEntry get() = catalog.models.first { it.kind == ModelKind.LLM && it.id != defaultLlm.id && it.status == app.murmur.core.catalog.Status.AVAILABLE && !it.noRussian }

    fun npuArtifact(e: ModelEntry) = e.artifacts.first { ComputeUnit.NPU in it.units }
    fun cpuArtifact(e: ModelEntry) = e.artifacts.first { it.units == listOf(ComputeUnit.CPU) }

    fun installed(ids: Collection<ArtifactId>): Map<ArtifactId, InstallState> = ids.associateWith { InstallState.Installed("/m/$it", 1) }
    fun installed(id: ArtifactId): Map<ArtifactId, InstallState> = installed(listOf(id))
    fun installed(a: ArtifactId, b: ArtifactId): Map<ArtifactId, InstallState> = installed(listOf(a, b))

    fun input(
        device: DeviceProfile? = emulator,
        states: Map<ArtifactId, InstallState> = emptyMap(),
        settings: Settings = Settings(),
        health: UnitHealth = UnitHealth.NONE,
        records: List<BenchmarkRecord> = emptyList(),
        storage: StorageInfo? = StorageInfo(2_400_000_000, 181_000_000_000),
        catalog: Catalog = this.catalog,
    ) = HubInput(catalog, states, settings, device, health, records.groupBy { it.artifactId }, storage)

    fun asrRecord(a: ArtifactId, unit: ComputeUnit, rtfx: Double, valid: Boolean = true, at: Long = 1_000) = BenchmarkRecord(
        a, unit, "0.1.0", "fp", at,
        BenchmarkResultData.Asr(rtfx, 400, 800, 120, 720, unit.name, emptyList(), valid, if (valid) null else "wer_high"),
    )
}
