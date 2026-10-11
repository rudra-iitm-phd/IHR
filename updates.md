## Current Changes


- integrating a constraint to penalize for violating traingle inequality
    - applied for `state_distance_metric` and 
    - `state_action_distance metric`

Generic form :
 - sample a batch $(z, c, r^c_z, z')\sim B$
 - constraint : $\text{ReLU}\bigg( d(sa, xb) - \underbrace{(d^-(sa, zc) + d^-( zc, xb))}_{\text{target metric}} \bigg)$

