package io.kinference.ndarray

import kotlinx.coroutines.CoroutineScope
import kotlin.coroutines.CoroutineContext

expect fun runBlocking(context: CoroutineContext, block: suspend CoroutineScope.() -> Unit)
