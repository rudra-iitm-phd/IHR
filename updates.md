## Current Changes


- Changed the policy loss. Fixed the issue with the relu and the predefined max value
- proportionately stretching the sate metric space to avoid representation collapse
- the `upper_bound_loss` in the `min_state_action_to_state_metric` function itself takes care of the equivalence loss
- current modifications included in the file `algo_cons3`
    - experiment results in log `algo_cons3_w_stretch`
- A second version of the loss is included in the file `algo_cons2`
    - Take grad steps with respect to the `state_distance` metric and find out the equivalent state `s_eq` with respect to `s`
    - use this state to transfer the skill in the policy
    - remarks - not tested
