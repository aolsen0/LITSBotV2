# LITSBotV2

LITSBotV2 is an engine to play the game Battle of LITS, intended to be an improvement over [a previous attempt](https://github.com/aolsen0/LITSBot).

## The game

Battle of LITS is an abstract strategy game developed by Grant Fikes in 2011. The rules can be found [here](https://www.nestorgames.com/rulebooks/BATTLEOFLITS_EN.pdf), and it can be played on Board Game Arena [here](https://boardgamearena.com/gamepanel?game=battleoflits). Some notable aspects of the game:

 - The number of available moves for most of the game is quite high, usually 100-150 during the middlegame. This presents a nice medium between the benchmark games of Chess and Go, but is enough that traversing the game tree at significant depth is quite difficult.
 - The games are quite short, with usually 12-16 pieces played on the board in a game. The result is that only a few moves are necessary for each player before the entire game tree can be explored, even with the high number of available moves.
 - The rules specify a tiebreaker (the player to play the last piece wins) and that the pie rule should be used, ensuring that the second player always wins with perfect play.

## Improvements over v1

 - More readable code.
 - Better performance in computing which moves are legal in any position. This is impactful, as this constitutes most of the computation time during play outside of running ML models for positional evaluation.
 - Instead of having a suite of models for positional evaluation, each trained to predict the best value of the previous one after one move, have a single RL model to predict the score changes for the entire remainder of the game. This is much more stable across turns and thus does much better at maing the first move with the pie rule in play.
 - Use torch.no_grad() during actual play.
 - Train for longer.