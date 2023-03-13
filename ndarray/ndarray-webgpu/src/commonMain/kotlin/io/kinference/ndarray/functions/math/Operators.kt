package io.kinference.ndarray.functions.math

class AddOperator : ArithmeticFunction() {
    override fun operation(input0: String, input1: String, output: String): String = "$output = $input0 + $input1;"
}

class SubOperator : ArithmeticFunction() {
    override fun operation(input0: String, input1: String, output: String): String = "$output = $input0 - $input1;"
}

class MulOperator : ArithmeticFunction() {
    override fun operation(input0: String, input1: String, output: String): String = "$output = $input0 * $input1;"
}

class DivOperator : ArithmeticFunction() {
    override fun operation(input0: String, input1: String, output: String): String = "$output = $input0 / $input1;"
}
