package io.kinference.algorithms.completion.model

import io.kinference.algorithms.completion.CompletionModels
import io.kinference.algorithms.completion.config.Config
import io.kinference.algorithms.completion.config.FilterConfig
import io.kinference.algorithms.completion.config.GenerationConfig
import io.kinference.algorithms.completion.suggest.*
import io.kinference.algorithms.completion.suggest.filtering.FilterModel
import io.kinference.algorithms.completion.suggest.filtering.ProbFilterModel
import io.kinference.algorithms.completion.suggest.ranking.FirstProbRankingModel
import org.junit.jupiter.api.Tag
import org.junit.jupiter.api.Test
import java.lang.System.currentTimeMillis

class FullModelPassTest {
    @Test
    @Tag("heavy")
    fun test() {
        val (tokenizerConfig, modelConfig, loader) = CompletionModels.v5

        val config = Config(loader, 10, tokenizerConfig, modelConfig, GenerationConfig.default, FilterConfig.default)

        val completionsCollector = FairseqCompletionsCollector(config)
        val rankingModel = FirstProbRankingModel()
        val filterModel = ProbFilterModel()
        val postFilterModel: FilterModel? = null

        val completionModel = CompletionModel(completionsCollector, rankingModel, filterModel, postFilterModel)

//        interaction(completionModel, config)
        speedTest(completionModel, config, 30, 20)
    }

    private fun interaction(completionModel: CompletionModel, config: Config) {
        println("Write something")
        var input = "hello wo"
        while (true) {
            val sepIndex = input.lastIndexOf(' ')
            val context = input.substring(0, sepIndex)
            val prefix = input.substring(sepIndex)
            println("$context:$prefix")

            val completions = completionModel.complete(context, prefix, config)
            println(completions)
            println()

            input = readLine().toString()
        }
    }

    private fun speedTest(completionModel: CompletionModel, config: Config, len: Int = 1, itersNum: Int = 100) {
//            1 - 0.97258
        println("Warm up")
        val input = (0 until len).map { "hello " }.joinToString { "" } + "wo"
//            val input = "hello wo"
        for (i in 0 until 10) {
            val sepIndex = input.lastIndexOf(' ')
            val context = input.substring(0, sepIndex)
            val prefix = input.substring(sepIndex)

            val completions = completionModel.complete(context, prefix, config)
        }

        println("Start")
        val startTime = currentTimeMillis()
        for (i in 0 until itersNum) {
            val sepIndex = input.lastIndexOf(' ')
            val context = input.substring(0, sepIndex)
            val prefix = input.substring(sepIndex)

            val completions = completionModel.complete(context, prefix, config)
        }
        val duration = (currentTimeMillis() - startTime) / 1000.0 / itersNum
        println()
        println(duration)
    }
}
