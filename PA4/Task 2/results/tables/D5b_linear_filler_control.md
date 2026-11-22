| Model | Seed | Acc % | Acc % with filler tokens not written |
|---|---|---|---|
| Linear (ELU+1) K4 mixed | 6304 | 28.8 | 28.6 |
| Linear (ELU+1) K4 mixed | 6305 | 28.4 | 28.3 |
| Linear (ELU+1) K4 mixed | 6306 | 27.6 | 27.5 |
| Linear (ELU+1) K8 mixed | 6304 | 53.4 | 53.2 |
| Linear (ELU+1) K8 mixed | 6305 | 6.3 | 6.3 |
| Linear (ELU+1) K8 mixed | 6306 | 53.4 | 53.3 |
| Linear (ELU+1) K16 mixed | 6304 | 10.2 | 10.1 |
| Linear (ELU+1) K16 mixed | 6305 | 99.7 | 99.6 |
| Linear (ELU+1) K16 mixed | 6306 | 6.5 | 6.5 |

ELU+1 linear attention, K = 16 test sequences: at test time filler positions contribute nothing to the state or the normaliser (both layers). If failure were interference from fillers, accuracy would rise.
