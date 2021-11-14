# legal-information-retrieval
Almost all our daily activities are regulated by the legislation of a superior institution, all these rules are generally written in natural language and quite hard to understand for an average citizen. Literacy in the legal domain is a subject of much research in Natural Language Processing, in this work, we study the ability of long-range Transformer --in our case BigBird--. to understand a quite long legal corpus. We trained two versions of BigBird. the first on Eurlex data -- publicly available legal corpus from the European Union, which mainly consists of statutes--, the second on the US legal cases from  [public legal cases](https://case.law/). After trained the two versions of BigBird on statutes and Legal case, we fine-tuned Legal-BigBird-legal-cases on jugement prior case identification and both Legal-Bigbird-case, and Legal-BigBird-statute on jugement statutes predictions, we compared the result of Legal-Bigbird against the results of the vanilla pretrained BigBird on general purpose copora. To fine tune we used the data set from [SigmaLaw](https://osf.io/76nmx/wiki/home/, http://ix.cs.uoregon.edu/~nisansa//DataSets.php)

# Pretrained Models on Hugingface
* [Legal-Bigbird-us](https://huggingface.co/lkwate/legal-bigbird-us)
* [Legal-Bigbird-eurlex](https://huggingface.co/lkwate/legal-bigbird-eurlex)

# Launch the MLM Training
```{sh}
python3 -m core.mlm_trainer
```

# Document Indexing
## Dense Indexing
The dense indexing is performed with our pretrained models
```{sh}
python3 -m core.index --encoder models/legal-bigbird-us --dimension 768 --corpus documents --index index/result-bigbird-us
```

# Use case of a most related case search
```{python}
searcher = EntailmentSearcher("models/legal-bigbird-us", "index/result-bigbird-us", "index/result-bm25", "documents/", top_k=1)
query = """plaintiff appellant john chen brought suit against major league baseball properties,and the office of the commissioner of baseball defendants alleging violations of the minimum wage and recordkeeping provisions of fair labor standards act flsa,et and the new york labor law nyll,et et.chen alleged that he worked without pay as a volunteer for fanfest,a interactive baseball theme park organized in conjunction with major league baseball all star week.defendants moved to dismiss the complaint asserting that fanfest is a seasonal amusement or recreational establishment and therefore exempt from the flsa minimum wage requirements pursuant to a by opinion and order the united states district court for the southern district of new york john koeltl,dismissed chen putative flsa collective action claims and declined to exercise supplemental jurisdiction over his nyll class action claims.chen appeals."""
out = searcher(query)
print(out)
```