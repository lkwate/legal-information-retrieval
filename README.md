# legal-information-retrieval
Almost all our daily activities are regulated by the legislation of a superior institution, all these rules are generally written in natural language and quite hard to understand for an average citizen. Literacy in the legal domain is a subject of much research in Natural Language Processing, in this work, we study the ability of long-range Transformer --in our case BigBird--. to understand a quite long legal corpus. We trained two versions of BigBird. the first on Eurlex data -- publicly available legal corpus from the European Union, which mainly consists of statutes--, the second on the US legal cases from[public legal cases](https://case.law/). After trained the two versions of BigBird on statutes and Legal case, we fine-tuned Legal-BigBird-legal-cases on jugement prior case identification and both Legal-Bigbird-case, and Legal-BigBird-statute on jugement statutes predictions, we compared the result of Legal-Bigbird against the results of the vanilla pretrained BigBird on general purpose copora. To fine tune we used the data set from [SigmaLaw](https://osf.io/76nmx/wiki/home/)

# sample data
```bash
python3 data-processing/data-sampling.py "data/AILA_2019/pair_statute.csv" "data/experiment/pair_statute" "label"
python3 data-processing/data-sampling.py "data/AILA_2019/pair_case.csv" "data/experiment/pair_case" "label"
```
