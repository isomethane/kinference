package io.kinference.graph

import io.kinference.data.ONNXData
import io.kinference.operator.*
import io.kinference.profiler.profile
import io.kinference.protobuf.message.*
import io.kinference.types.ValueInfo
import io.kinference.utils.*
import kotlinx.coroutines.coroutineScope
import kotlin.time.ExperimentalTime

//TODO: check i/o tensor shapes explicitly
//TODO: graph optimizations (i.e. remove "Identity" nodes, fuse "MatMul" with "Add" etc)
@ExperimentalTime
abstract class Graph<T : ONNXData<*, *>> protected constructor(
    proto: GraphProto,
    private var _operators: ArrayList<Operator<T, T>>,
    private val valueOrderInfo: GraphValueOrderInfo
) : Closeable {
    companion object {
        private val logger = LoggerFactory.create("io.kinference.core.graph.Graph")

        private fun GraphValueOrderInfo.putOrderFor(names: Set<String>, order: Int, initNames: List<String>) {
            val (_, otherNames) = names.partition { name -> initNames.any { it == name } }
            putOrder(otherNames, order)
        }

        fun <T : ONNXData<*, *>> GraphProto.collectOperators(valueOrderInfo: GraphValueOrderInfo): ArrayList<Node> {
            val sortedNodes = ArrayList<Node>(this.node.size)
            val initNames = initializer.map { it.name ?: "" }
            val nodes = HashMap<String, Node>().apply {
                for (nodeProto in node) {
                    val node = Node(nodeProto)
                    for (output in nodeProto.output) {
                        put(output, node)
                    }
                }
            }

            val stack = Stack<Node>().apply {
                for (output in output) {
                    val name = output.name!!
                    if (name.isNotEmpty()) {
                        val node = nodes[name]
                        if (node != null) push(node)
                    }
                }
            }

            var order = 0
            val outputNames = this.output.map { it.name!! }
            while (stack.isNotEmpty()) {
                val node = stack.peek()
                if (!node.visited) {
                    var ready = true
                    for (input in node.dependencies) {
                        val next = nodes[input]
                        if (next != null && !next.visited) {
                            ready = false
                            stack.push(next)
                        }
                    }

                    if (ready) {
                        node.visited = true
                        stack.pop()
                        sortedNodes.add(node)
                        valueOrderInfo.putOrderFor(node.dependencies - outputNames, order, initNames)
                        order++
                    }
                } else stack.pop()
            }

            return sortedNodes
        }
    }
    val operators: List<Operator<T, T>>
        get() = _operators

    val inputs = proto.input.map { ValueInfo.create(it) }
    val outputs = proto.output.map { ValueInfo.create(it) }
    val info = proto.valueInfo.map { ValueInfo.create(it) }

    protected val initializers = ArrayList<T>(proto.initializer.size).apply {
        for (i in proto.initializer)
            this.add(prepareInput(i))
    }

    val initNames = proto.initializer.map { it.name }

    data class Node(val proto: NodeProto, var visited: Boolean = false) {
        private fun NodeProto.collectRequiredInputs(): Set<String> = HashSet<String>().apply {
            for (variable in input) {
                if (variable.isNotEmpty()) add(variable)
            }

            for (attr in attribute) {
                if (attr.type == AttributeProto.AttributeType.GRAPH) {
                    val subGraphInputs: HashSet<String> = attr.g!!.input.mapTo(HashSet()) { it.name!! }

                    val subGraphLocalInputs = attr.g!!.node.flatMapTo(HashSet()) { it.collectRequiredInputs() }
                    subGraphInputs.addAll(attr.g!!.output.map { it.name!! })

                    val subGraphLocalOutputs = attr.g!!.node.flatMapTo(HashSet()) { it.output }

                    addAll((subGraphLocalInputs - subGraphLocalOutputs) - subGraphInputs)
                }
                // TODO AttributeProto.AttributeType.GRAPHS
            }
        }

        val dependencies by lazy { proto.collectRequiredInputs() }
    }

    init {
        if (_operators.size != proto.node.size) {
            logger.warning {
                "Count of used operators ${operators.size} not equals count of operators in model ${proto.node.size}. " +
                    "Remove unused operators from model for more performance!"
            }
        }
    }

    abstract fun prepareInput(proto: TensorProto): T

    fun addInitializer(initializer: T) {
        require(!initializers.any { it.name == initializer.name }) { "Initializer with name ${initializer.name} already exists" }
        initializers.add(initializer)
        valueOrderInfo.putOrder(initializer.name!!, Int.MAX_VALUE)
    }

    fun findInitializer(name: String): T? {
        return initializers.find { it.name == name }
    }

    fun mergeOperators(names: List<String>, to: Operator<T, T>) {
        fun MutableList<Operator<T, T>>.removeOperator(i: Int) {
            this.removeAt(i)
            valueOrderInfo.decOrderFrom(i)
        }

        fun MutableList<Operator<T, T>>.addOperator(i: Int, op: Operator<T, T>) {
            this.add(i, op)
            valueOrderInfo.incOrderFrom(i)
        }

        val namesToRemove = names.toHashSet()
        val newOperators = ArrayList(_operators)
        var lastIdx = -1
        val toRemove = ArrayList<Int>(0)
        for (name in namesToRemove) {
            val i = newOperators.indexOfFirst { it.name == name }
            if (i == -1) error("Cannot remove $name operator. Operator $name was not found")
            if (lastIdx < i) lastIdx = i
            toRemove.add(i)
        }
        for (op in toRemove.sortedDescending()) newOperators.removeOperator(op)
        newOperators.addOperator(lastIdx - namesToRemove.size + 1, to)

        _operators = newOperators
    }

    private fun GraphValueOrderInfo.decOrderFrom(targetOrder: Int) {
        for (name in this.names()) {
            val order = valueOrderInfo.getOrder(name)
            if (order >= targetOrder) valueOrderInfo.putOrder(name, order - 1)
        }
    }

    private fun GraphValueOrderInfo.incOrderFrom(targetOrder: Int) {
        for (name in this.names()) {
            val order = valueOrderInfo.getOrder(name)
            if (order <= targetOrder) valueOrderInfo.putOrder(name, order + 1)
        }
    }

    val availableInputs: List<String> = inputs.map { it.name }

    private fun cleanupUntilOrder(context: GraphContext<T>, order: Int) {
        context.removeValues { it !in availableInputs && valueOrderInfo.getOrder(it) <= order }
    }

    protected abstract fun makeContext(root: GraphContext<T>?): GraphContext<T>

    override fun close() {
        closeAll(initializers)
        closeAll(operators)
    }

    @ExperimentalTime
    suspend fun execute(inputs: List<T>, _contexts: Contexts<T> = emptyContexts()): List<T> {
        //TODO: check that all inputs were set and not null
        val contexts = Contexts(makeContext(_contexts.graph), _contexts.profiling)

        for (tensor in initializers) {
            contexts.graph!!.putValue(tensor.name!!, tensor)
        }

        for (input in inputs) {
            if (input.name !in availableInputs) {
                logger.warning { "Input node '${input.name}' not found in Graph and probably is excessive" }
                continue
            }
            contexts.graph!!.putValue(input.name!!, input)
        }

        coroutineScope {
            for ((i, operator) in operators.withIndex()) {
                lateinit var outputs: List<T?>
                contexts.profiling.profile(operator.info.type) { profilingContext ->
                    outputs = operator.applyWithCheck(
                        Contexts(contexts.graph, profilingContext),
                        operator.inputs.map { input -> if (input.isEmpty()) null else contexts.graph!!.getValue(input) })
                }

                contexts.profiling.profile("${operator.info.type}:cleanup") {
                    cleanupUntilOrder(contexts.graph!!, i)
                    for (outputIdx in outputs.indices) {
                        val output = outputs[outputIdx]
                        val variable = operator.outputs.getOrNull(outputIdx)

                        if (output == null) require(variable.isNullOrEmpty()) { "Required output '$variable' not provided by '${operator.info.type}' operator" }
                        if (!variable.isNullOrEmpty()) {
                            contexts.graph.putValue(variable, output!!.rename(name = variable) as T)
                        } else {
                            output?.close()
                        }
                    }
                }
            }
        }
        return outputs.map { contexts.graph!!.getValue(it.name) }
    }
}
