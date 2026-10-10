## Current Changes

- Soft Contrastive loss reimplemented in the `actor_loss` section. 
- Tried to maximize diversity by penalizing low variance in the metric space. Doesn't work that well. Commented out for the time being
- Upperbound integrated for the `min_state_action_to_state_distance`. Formalized as the following $$\min_h \mathbb{E}_{(s, a), x \sim B}\bigg[\text{ReLU}\bigg(h(d)((s, a), x) -  \Delta(d)((s, a), (x, a))\bigg)\bigg]$$
- Policy vaidation steps added
    - In particular we want to validate whether policy is using the information of the metric space to actually take actions.
    - Log the state pairs with high similarity and check if the actions taken by the policy in the corresponding states are similar as well
- check the results from the logs `algo_cons_iter4`
