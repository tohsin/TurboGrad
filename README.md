# TurboGrad
 Called turbo not because its fast but because its a mini torch, sorta. Repo was adapted from andrej kaparthy, but modified to solve RL tasks providing an overall framework for people to learn both RL and pytorch and how to implement these algorithms, It also includes many other features not included in Andrej work like other optimisers (ADAM, RMSProp) and detach functions etc. Have fun and dont forget to call Zero_grad before backward.

 Main modification required now is to figure out vecorisation with the Value class in a list, will serve as major speed up for the library.