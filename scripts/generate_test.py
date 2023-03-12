import math
from itertools import product
import torch

torch.manual_seed(0)

s0, s1, s2 = 2, 3, 4

inputs = torch.rand(s0)
label = torch.rand(s2)

w1 = torch.rand((s1, s0), requires_grad=True)
w2 = torch.rand((s2, s1), requires_grad=True)
b1 = torch.rand(s1, requires_grad=True)
b2 = torch.rand(s2, requires_grad=True)

outputs = torch.sigmoid(w2 @ torch.sigmoid(w1 @ inputs + b1) + b2)
loss = (outputs - label).square().sum()

loss.backward()

# print("grads")
# print(f"{w1.grad=}")
# print(f"{b1.grad=}")
# print(f"{w2.grad=}")
# print(f"{b2.grad=}")

make_network = f"Network network = make_network({s0}, {s1}, {s2});"

square_brackets = lambda indices, shape: " + ".join(
    f"{idx} * {math.prod(shape[dim + 1:])}" for (dim, idx) in enumerate(indices)
)

fill_array = lambda name, tensor: "\n".join(
    f"{name}[{square_brackets(indices, tensor.shape)}] = {tensor[indices]};"
    for indices in product(*map(range, tensor.shape))
)

alloc_array = (
    lambda name, tensor: f"double *{name} = malloc({tensor.numel()} * sizeof(double));"
)

gradients = [
    ("nabla_w1", w1.grad),
    ("nabla_b1", b1.grad),
    ("nabla_w2", w2.grad),
    ("nabla_b2", b2.grad),
]

back_prop = """\
double *nabla_w[] = {
    malloc(network.s0 * network.s1 * sizeof(double)),
    malloc(network.s1 * network.s2 * sizeof(double)),
};
double *nabla_b[] = {
    malloc(network.s1 * sizeof(double)),
    malloc(network.s2 * sizeof(double)),
};
double loss = back_propagation(network, inputs, label, nabla_w, nabla_b);
"""

code = "\n".join(
    [
        make_network,
        "\n// fill network",
        *(
            fill_array(f"network.{name}", tensor)
            for (name, tensor) in [("w1", w1), ("w2", w2), ("b1", b1), ("b2", b2)]
        ),
        "\n// fill gradients",
        *(
            alloc_array(name, tensor) + "\n" + fill_array(name, tensor)
            for (name, tensor) in gradients
        ),
        "\n// fill inputs and label",
        *(
            alloc_array(name, tensor) + "\n" + fill_array(name, tensor)
            for (name, tensor) in [("inputs", inputs), ("label", label)]
        ),
        "\n// run backprop",
        back_prop,
        "// compare loss",
        f'assert_scalar("loss", {float(loss)}, loss);',
        "\n// compare gradients",
        *(
            f'assert_array("{name}", {tensor.numel()}, {name}, {name[:-1]}[{int(name[-1]) - 1}]);'
            for (name, tensor) in gradients
        ),
        "\n// free gradients",
        *(f"free({name});" for (name, _) in gradients),
        *(
            f"free({name}[{i}]);"
            for (name, i) in product(["nabla_w", "nabla_b"], range(2))
        ),
        "// free inputs and labels",
        *(f"free({name});" for name in ["inputs", "label"]),
    ]
)

indent = lambda text: "    " + text.replace("\n", "\n    ")
remove_trailing_whitespace = lambda text: "\n".join(map(str.rstrip, text.split("\n")))

print("/* automatically generated by 'generate_test.py' */")
print("void test_back_propagation()\n{")
print(remove_trailing_whitespace(indent(code)))
print("}")
