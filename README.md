# micppo: Another Proximal Policy Optimization (PPO) implementation
This is my implementation of Proximal Policy Optimization.

I wrote this code following [a tutorial from Costa Huang](https://www.youtube.com/watch?v=MEt6rrxH8W4).

## Installation
Clone the repository, cd into it and then run

`pip install .`

The command above will install the `micppo` script into your PATH.

## Usage
The `micppo` script allows to make experiments with the
[gymnasium](https://gymnasium.farama.org/) environments. By default it
runs on "CartPole", but you can configure it. It logs results for tensorboard.
Run

`micppo --help`

for having a complete command-line help.

## Author
Michele De Stefano
