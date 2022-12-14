# Equality Transformer

This code is based on a project for the graduate-level course "Advanced Formal Language Theory" by Prof. Dr. Ryan Cotterell at ETH Zurich by Aymeric Ducatez, Alexis Perakis, and Arman Raayatsanati extending the paper "Overcoming a Theoretical Limitation of Self-Attention" by David Chiang and Peter Cholak in _Proc. ACL_, 2022. Parts of this code were inspired by the GitHub repository of the paper (https://github.com/ndnlp/parity).

We test our approach on the formal language EQUALITY. To reproduce the results, any of the equality files can be run using Python.

equality_exact.py shows our exact solution.
equality_exact_layernorm.py shows our exact solution with layer normalisation.
equality.py and equality_old.py show the training at fixed string length with and without position encoding.