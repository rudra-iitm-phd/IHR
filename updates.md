## Current Changes


- Upperbound integrated for the `min_state_action_to_state_distance`. Formalized as the following $$\min_h \mathbb{E}_{(s, a), x \sim B}\bigg[\text{ReLU}\bigg(h(d)((s, a), x) -  \Delta(d)((s, a), (x, a))\bigg)\bigg]$$ and as well as 
$$\min_h \mathbb{E}_{(s, a), x \sim B}\bigg[\text{ReLU}\bigg(h(d)((s, a), x) -  d(s, x)\bigg)\bigg]$$

- check the results from the logs `algo_cons_iter5`

