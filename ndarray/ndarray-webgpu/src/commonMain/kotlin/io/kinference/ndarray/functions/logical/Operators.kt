package io.kinference.ndarray.functions.logical

class EqualOperator : LogicalOperator() {
    override fun operation(input0: String, input1: String, output: String): String = "$output = i32($input0 == $input1);"
}

class GreaterOperator : LogicalOperator() {
    override fun operation(input0: String, input1: String, output: String): String = "$output = i32($input0 > $input1);"
}

class LessOperator : LogicalOperator() {
    override fun operation(input0: String, input1: String, output: String): String = "$output = i32($input0 < $input1);"
}

class OrOperator : LogicalOperator() {
    override fun operation(input0: String, input1: String, output: String): String = "$output = i32(bool($input0) || bool($input1));"
}
