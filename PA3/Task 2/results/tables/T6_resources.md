**Compute and memory (Kaggle T4). Training time excludes development evaluation and checkpoint I/O; feature caching is separate from training; peak = max allocated GPU memory.**

| Phase | Updates | Trainable params | Time (min) | Peak GPU (GB) | Notes |
|---|---|---|---|---|---|
| Feature caching (SigLIP, frozen) | -- | 0 | 3.3 | 0.89 | 9272 images, 2662 MB FP16 |
| 196 tokens (main): caption_alignment | 2250 | 1,837,056 | 44.9 | 1.56 | 1.20 s/update |
| 196 tokens (main): visual_instruction | 1500 | 6,424,576 | 36.7 | 1.55 | 1.47 s/update |
| 196 tokens (main): vqa_specialization | 750 | 6,424,576 | 18.1 | 1.55 | 1.45 s/update |
| 196 tokens (main): dev evaluation (12 points) | -- | -- | 12.7 | 1.57 | checkpoint I/O 0.0 min |
| 196 tokens (main): final evaluation (2 suites x 4 conditions) | -- | -- | 2.4 | 1.49 | 33.2 ms/answer (greedy, batch 8) |
| 49 tokens: caption_alignment | 2250 | 1,837,056 | 19.3 | 1.48 | 0.52 s/update |
| 49 tokens: visual_instruction | 1500 | 6,424,576 | 19.4 | 1.48 | 0.78 s/update |
| 49 tokens: vqa_specialization | 750 | 6,424,576 | 9.6 | 1.42 | 0.77 s/update |
| 49 tokens: dev evaluation (12 points) | -- | -- | 11.7 | 1.39 | checkpoint I/O 0.0 min |
| 49 tokens: final evaluation (2 suites x 4 conditions) | -- | -- | 2.3 | 1.26 | 24.2 ms/answer (greedy, batch 8) |
