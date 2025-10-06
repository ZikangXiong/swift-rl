# Swift-RL: Fast and Robust Reinforcement Learning
# Implementation of Swift-TD and Swift-SARSA algorithms

__version__ = "0.1.0"

from swift_rl.policies import epsilon_greedy, softmax_policy
from swift_rl.swift_sarsa import SwiftSARSA
from swift_rl.swift_td import SwiftTD

__all__ = ["SwiftTD", "SwiftSARSA", "epsilon_greedy", "softmax_policy"]
