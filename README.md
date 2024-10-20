# Rust NEAT
*A matura Thesis*

Simple implementation of NEAT (Neuroevolution of Augmented Topologies) applied on games like Nim, Tic Tac Toe etc.\
You can find the newest pdf version of the thesis [here](/out/ma.pdf).

## Games and inputs:
Simple Nim & Nim 
Input – binary for all existing matches
Output – the amount to take away for any pile

## Tic Tac Toe
Input – One hot encoding of game (three states per field – x/o/empty)
Output – placement field of next x/o

## Fitness
Create a batch and let them play each other in multiple games against different opponents
Fitness -> primarily number of won games
Objective evaluation by a perfect algorithm with solved games (not the evolutionarily determining factor)

## 1.	Prove of Concept and find out strategies/parameters for mutation and competition
Idea:
Create Neural Network that solves Simple (and Normal) Nim with Neuroevolution
Build a FNN (feed forward Neural network) and train them with reinforcement learning as a prove that a network this size can do the task
Then replicate the network multiple times with random weights and mutate their weights over generations
No Genes but only weight mutation, No mating, no adding of nodes, weight mutation proportional to their previous weight for continuity (range ex. +/- 20% of previous weight), only few weights modified per mutation for continuity
Simple Competition: Population competes then the best fraction (ex ¼) survives and creates 1/fraction offspring

## 2.	(Future) Research and complexification
Condition: Working base model established 
Research about more complex strategies, specifically NEAT (Neuroevolution of Augmented Topologies) and implementation
Test with the same games (and harder games)
