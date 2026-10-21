| Model | E(5) 1× | C_M | E_M | C_R | E_R | E(2) 1× | E(2) 2× | E(2) 3× | Params | Train s/epoch | Inference s |
|---|---|---|---|---|---|---|---|---|---|---|---|
| sym. B | 0.0211 (0.0217, 0.0205) | 8.56e-17 (7.95e-17, 9.17e-17) | 0.0211 (0.0217, 0.0205) | 6.89e-17 (6.42e-17, 7.36e-17) | 0.0211 (0.0217, 0.0205) | 0.0178 (0.0183, 0.0173) | 0.0427 (0.0419, 0.0434) | 0.0460 (0.0448, 0.0472) | 480,784 | 11.5 (11.5, 11.5) | 3.40 (3.39, 3.41) |
| sym. D | 0.0273 (0.0272, 0.0273) | 6.12e-17 (6.02e-17, 6.21e-17) | 0.0273 (0.0272, 0.0273) | 4.71e-17 (4.62e-17, 4.80e-17) | 0.0273 (0.0272, 0.0273) | 0.0246 (0.0246, 0.0246) | 0.0290 (0.0287, 0.0293) | 0.0327 (0.0324, 0.0329) | 480,784 | 16.3 (16.3, 16.3) | 5.57 (5.61, 5.53) |

Test split. Symmetrised models (eq. 7). Training time is that of the underlying learned model. Timings on training: Tesla T4 (15 GB), torch 2.11.0+cu130 | inference: Tesla T4 (15 GB), torch 2.11.0+cu130. Inference includes all 8 transformations, forward passes, inverse transformations and the averaging.
