# Neural

This repo contains a simple implementation of a neural network + backpropagation written in C.

## Installation

You can run `neural` using [`Nix`](https://zero-to-nix.com/):

```
nix run github:tiny-technologies/neural
```

Or, install it:

```
nix profile install github:tiny-technologies/neural
```

## Usage

Here is a summary of the available commands and their respective arguments:

### Run

Predict the label of an image using a trained network:

```
neural run <path_to_model> <path_to_image>
```

The image needs to be in the [PGM P5](https://en.wikipedia.org/wiki/Netpbm) format. You can edit the `example.pgm` with an image manipulation tool of your choice (e.g. GIMP).

### Test

Test the accuracy of a trained network on the `test data`:

```
neural test <path_to_model>
```

### Train

To finetune an existing model or train a new one from scratch, use the

```
neural train
```

subcommand. For detailed list of command line flags see below.

### Help

Show all subcommands and their respective options:

```
neural help
```

```
Usage:

    neural <command> [<args>]

Commands:

    run    Run inference using a trained network
      <path>                      path to model
      <path>                      path to PGM P5 image 

    test   Test the accurary of a trained network
      <path>                      path to model (default: default.model)

    train  Train a new network and store it to disk
      -b, --batch-size <int>      samples per batch (default: 200)
      -d, --dims <int,int,..>     dimensions of hidden layers (default: 16,16)
      -e, --epochs <int>          number of epochs (default: 10)
      -l, --learning-rate <real>  step size of parameter update (default: 0.01)
      -i, --input <path>          path to model used as starting point (optional)
      -o, --output <path>         output path of the trained model (default: default.model)

    bench  Benchmark forward and backward pass

    help   Show this message and exit

```

## Development

### Tooling

When you `cd` into this repository, [`direnv`](https://direnv.net/) activates a [`Nix`](https://zero-to-nix.com/)-environment containing all required development tools.

### Workflow

The [`just`](https://github.com/casey/just) command is the main entry point into this repository. It is a convenient way to execute commands and has a documenting purpose by making important project-specific commands visible.

To list all commands, run:

```sh
just
```

Download the MNIST dataset:

```sh
just download
```

Compile and run the debug build:

```sh
just debug
```

Compile and run the release build:

```sh
just release
```

Compile and run tests:

```sh
just test
```

Compile in release mode, with optimizations:

```sh
just build release
```
