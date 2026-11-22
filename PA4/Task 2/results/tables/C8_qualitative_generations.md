| Prompt | Gold | FP16 full cache | Evict: sink+recent | Evict: question-aware | INT4 channel-K/token-V | INT4 token-wise K/V | Hybrid FP16 + INT2 |
|---|---|---|---|---|---|---|---|
| KV-2048-start-8-00 | 67030 | '67030' | '74262' | '67030' | '67030' | '7 warehouse 67 warehouse 77 warehouse 4 | '67030' |
| KV-2048-middle-8-00 | 84561 | '84561' | '20488' | '84561' | '84561' | ' airport  river  river river river rive | '84561' |
| KV-2048-end-8-00 | 33634 | '33634' | '33634' | '33634' | '33634' | ' airport 448 parcels 88 parcels 4 wareh | '33634' |
| KV-4096-start-8-00 | 21406 | '21406' | '15576' | '21406' | '21406' | ' 6 river   river river  river   river r | '21406' |
| KV-4096-middle-8-00 | 73510 | '73510' | '62526' | '73510' | '73510' | '  river  river warehouse  river  river  | '73510' |
| KV-4096-end-8-00 | 59145 | '59145' | '59145' | '59145' | '59145' | '                       4        ' | '59145' |

Fixed prompts chosen before results (config part_c.qualitative_ids); masks in figures/FC3.
