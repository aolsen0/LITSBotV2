# LITSBotV2

LITSBotV2 is an engine to play the game Battle of LITS, intended to be an improvement over [a previous attempt](https://github.com/aolsen0/LITSBot).

## The game

Battle of LITS is an abstract strategy game developed by Grant Fikes in 2011. The rules can be found [here](https://www.nestorgames.com/rulebooks/BATTLEOFLITS_EN.pdf), and it can be played on Board Game Arena [here](https://boardgamearena.com/gamepanel?game=battleoflits). Some notable aspects of the game:

 - The number of available moves for most of the game is quite high, usually 100-150 during the middlegame. This presents a nice medium between the benchmark games of Chess and Go, but is enough that traversing the game tree at significant depth is quite difficult.
 - The games are quite short, with usually 12-16 pieces played on the board in a game. The result is that only a few moves are necessary for each player before the entire game tree can be explored, even with the high number of available moves.
 - The rules specify a tiebreaker (the player to play the last piece wins) and that the pie rule should be used, ensuring that the second player always wins with perfect play.

## Improvements over v1

 - More readable code
 - A single model for positional evaluation, trained to predict the highest/lowest evaluations (by itself) after 1 move. I hope this will work
 - Use torch.no_grad() during actual play
 - Train for longer
 - C++ implementation of the game tree search. It is probably possible to do this efficiently enough to make the positional evaluation model obsolete.