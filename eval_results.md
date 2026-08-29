# StyleSense retrieval ablation

16 questions, 16 configurations, `text-embedding-3-small`.
Coverage is the percentage of expected keywords appearing in the retrieved chunks. Ceiling is the best coverage arithmetically possible at that k.

```
chunks        k      search     MRR    nDCG  coverage  ceiling   fact    occ    con   enum
------------------------------------------------------------------------------------------
c2000_o400   16         mmr   0.412   0.491     75.6%   100.0%    90%    59%    70%    81%
c2000_o400   16  similarity   0.399   0.476     74.9%   100.0%   100%    54%    60%    81%
c1000_o200   16  similarity   0.358   0.436     72.7%   100.0%   100%    47%    58%    81%
c2000_o400    8  similarity   0.394   0.463     69.6%    90.8%   100%    42%    58%    71%
c500_o100    16  similarity   0.311   0.393     66.7%   100.0%    93%    30%    58%    83%
c1000_o200   16         mmr   0.354   0.418     65.2%   100.0%    90%    39%    58%    69%
c1000_o200    8  similarity   0.351   0.417     64.0%    90.8%    90%    32%    58%    71%
c500_o100     8  similarity   0.303   0.368     57.4%    90.8%    83%    25%    54%    62%
c500_o100    16         mmr   0.297   0.353     55.6%   100.0%    90%    27%    49%    45%
c1000_o200    8         mmr   0.344   0.390     53.6%    90.8%    90%    27%    41%    45%
c2000_o400    4  similarity   0.362   0.398     49.8%    72.3%    90%    30%    27%    40%
c2000_o400    8         mmr   0.339   0.374     48.3%    90.8%    60%    38%    54%    36%
c1000_o200    4  similarity   0.324   0.359     47.8%    72.3%    90%    25%    27%    36%
c500_o100     4  similarity   0.277   0.308     39.8%    72.3%    67%    19%    38%    26%
c500_o100     8         mmr   0.254   0.276     36.0%    90.8%    60%    18%    35%    21%
c2000_o400    4         mmr   0.296   0.306     33.4%    72.3%    60%    16%    27%    21%
c1000_o200    4         mmr   0.252   0.257     27.1%    72.3%    53%     9%    21%    17%
c500_o100     4         mmr   0.229   0.239     26.8%    72.3%    53%    11%    23%    10%
```
