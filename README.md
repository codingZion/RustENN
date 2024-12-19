# Rust ENN
*A Matura Thesis*

Simple implementation of Evolutionary Neural Networks (ENNs) inspired by the NEAT algorithm applied on games.\
You can find the newest pdf version of the thesis [here](/thesis/ma.pdf).


**Abstract.**  
This thesis explores the design and implementation of evolutionary
neural networks (ENNs) inspired by the NEAT algorithm. ENNs optimize
neural network weights, biases, and topologies using evolutionary
computation. The game Nim and a simplified version with only one instead
of multiple stacks are used as a benchmark of the ENN algorithm to test
the effect of different parameters and features on performance. The
research shows that ENNs can successfully learn to play simple games
like single-stack Nim but struggle with more complex games like
multi-stack Nim. Challenges include overcoming local minima, tuning the
ENN parameters correctly, and dealing with limited computations.  
**Results.** The results show that ENNs achieve 100% accuracy for the
single-stacked version of the game, with stack sizes up to 1000 matches,
revealing the importance of using correct output encoding. In the
multi-stacked Nim game, the ENNs reach an accuracy of 100% for the
configuration with 2 stacks containing 2 matches each and 75% for the
configuration with 2 stacks containing 4 matches each. In the latter
configuration, the NNs could not predict the perfect moves for all game
states because of local minima.  
**Keywords.** Evolutionary Neural Networks, NEAT Algorithm, Machine
Learning, Game Learning, Nim Game, Evolutionary Computation.

**Tools used during Research**\
To complete this research, following tools were used:

-   [**Rust**](https://www.rust-lang.org/) as the programming language.

-   [**Rust Crates**](https://crates.io/), [rand (version 0.9.0-alpha.2)](https://crates.io/crates/rand/0.9.0-alpha.2), [indicatif (version 0.17.8)](https://crates.io/crates/indicatif/0.17.8), [csv (version 1.3.0)](https://crates.io/crates/csv/1.3.0), [bincode (version 1.3.3)](https://crates.io/crates/bincode/1.3.3), [serde (version 1.0.209)](https://crates.io/crates/serde/1.0.209), and [rayon (version 1.10.0)](https://crates.io/crates/rayon/1.10.0).

-   [**JetBrains RustRover**](https://www.jetbrains.com/rustrover/) as the integrated development environment.

-   [**Git**](https://git-scm.com/) as the version control system.

-   [**GitHub**](https://github.com/) as the online host of the repository.

-   [**GitHub Copilot**](https://github.com/features/copilot/) as the AI code completion tool to accelerate programming.

-   [**LaTeX**](https://www.latex-project.org/) as the typesetting system for the thesis.

-   [**OpenAI ChatGPT**](https://openai.com/chatgpt/) to enhance the language for some pre-written text parts and help troubleshoot technical problems.

-   [**Phoenix GUI**](https://github.com/TomtheCoder2/phoenix_gui) as a graphing tool to visualize the results of the tests.

A big thank goes also to Gerd Altmann, for providing the [**cover page image**](https://pixabay.com/de/illustrations/ai-generiert-neuronen-gehirnzellen-9022566/) of this thesis.
