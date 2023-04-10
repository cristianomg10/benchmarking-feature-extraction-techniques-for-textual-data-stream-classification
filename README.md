# Benchmarking Feature Extraction Techniques for Textual Data Stream Classification

This repo regards the paper [1]. In this paper, we evaluated three four representation approaches, i.e., BERT[2], Hashing Trick [3], Word2Vec [4], and Incremental Word-Vectors [5]. These approaches were evaluated considering accuracy and run time, considering different stream lengths, i.e., 10k, 20k, 30k, 50k, 100k, 200k. Also, two datasets were used in the evaluation: Sentiment140 and Yelp dataset. The details on the processing are described in the original paper.

The results are stored in [this file](https://docs.google.com/spreadsheets/d/1S2RERdbW9Cxt-GwlcmeAPPtxsxjDDhJfV13Nh-KfAUc/edit?usp=sharing).

[1] Thuma, B. S, Vargas, P. S., Garcia, C. M., Britto Jr, Alceu de, & Barddal, Jean P. (2023). Benchmarking Feature Extraction Techniques for Textual Data Stream Classification (*Accepted for presentation at the International Joint Conference on Neural Networks*). 

[2] Devlin, J., Chang, M. W., Lee, K., & Toutanova, K. (2018). BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding. arXiv preprint arXiv:1810.04805.

[3] Attenberg, J., Weinberger, K., Dasgupta, A., Smola, A., & Zinkevich, M. (2009, July). Collaborative Email-spam Filtering with the Hashing Trick. In Proceedings of the Sixth Conference on Email and Anti-spam.

[4] Mikolov, T., Sutskever, I., Chen, K., Corrado, G. S., & Dean, J. (2013). Distributed Representations of Words and Phrases and Their Compositionality. Advances in Neural Information Processing Systems, 26.

[5] Bravo-Marquez, F., Khanchandani, A., & Pfahringer, B. (2022). Incremental Word Vectors for Time-evolving Sentiment Lexicon Induction. Cognitive Computation, 1-17.