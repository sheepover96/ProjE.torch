# ProjE.torch

pytorch implementation of ProjE: Embedding Projection for Knowledge Graph Completion.
https://arxiv.org/abs/1611.05425

# How to run code

```
pip install -r requirements
python main.py --vector_dim 200 --sample_p 0.5 --batch_size 200 --nepoch 100 --batch_size 200 --lr 0.01 --alpha 1e-5 --loss_method wlistwise
```