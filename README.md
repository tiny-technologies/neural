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

```
Usage:

    neural <command> [<args>]

These are common commands:

    run    Run inference using a trained network
    train  Train a new network and store it to disk
    test   Run unit tests
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
