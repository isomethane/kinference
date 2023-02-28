package io.kinference.utils

import okio.Path
import okio.Path.Companion.toPath

interface TestDataLoader : DataLoader {
    fun getFullPath(path: Path): Path
}

object ResourcesTestDataLoader : TestDataLoader {
    private val mainPath =
        ("${PlatformUtils.forPlatform("", "../../utils/test-utils/")}build/processedResources/" +
        "${PlatformUtils.forPlatform("jsLegacy", "jvm")}/main/").toPath()

    override fun getFullPath(path: Path) = mainPath / path

    override suspend fun bytes(path: Path): ByteArray = CommonDataLoader.bytes(mainPath / path)
    override suspend fun text(path: Path): String = CommonDataLoader.text(mainPath / path)
}

object S3TestDataLoader : TestDataLoader {
    private val mainPath: Path =
        PlatformUtils.forPlatform("s3", "../../test-data/s3/tests").toPath()

    override fun getFullPath(path: Path) = mainPath / path

    override suspend fun bytes(path: Path): ByteArray = CommonDataLoader.bytes(mainPath / path)
    override suspend fun text(path: Path): String = CommonDataLoader.text(mainPath / path)
}
