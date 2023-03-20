<h1> Comparative study of Masked Language Modeling and Fine-tuning BERT  </h1>

This repository contains BERT models - BERT + NN, BERT + Attention, BERT + CNN

Experiements 
1. GLUE datasets - MRPC, WNLI, RTE, COLA.
2. BERT models - BERT + NN, BERT + Attention, BERT + CNN.
3. Tasks - Masked Language Modeling and Fine-tuning using downstream task data
 
Datasets:
1. MRPC (Microsoft Research Paraphrase Corpus): This dataset consists of sentence pairs labeled as either paraphrases or non-paraphrases. It contains 5,801 sentence pairs and is derived from news articles, web pages, and other sources.

2. WNLI (Winograd NLI): This dataset is based on the Winograd Schema Challenge and consists of 634 sentence pairs. The goal is to determine if the second sentence can be inferred from the first sentence based on the context.

3. RTE (Recognizing Textual Entailment): This dataset consists of sentence pairs labeled as either entailment, contradiction, or neutral. It contains 2,490 sentence pairs and is derived from news articles, fiction, and other sources.

4. COLA (Corpus of Linguistic Acceptability): This dataset consists of sentences labeled as either grammatically correct or incorrect. It contains 10,000 sentences and is derived from sources such as books, news articles, and websites.

Result:
Increase of ~8% on average evaluation metric (f1 for MRPC and accuracy for other datasets) value on 4 GLUE datasets. 
Results are present in src/results/BERT_experiments.xlsx file
