# CUPT Source Code Sample

*Simulation for Problem 7 Conical Piles*

The dynamics of sandpile is considered as a systematic event emerge from interactions among massive sand particles, which could be modeled by Cellular Automaton.



**Released at 2018.4**

This is a demonstration project for China Undergrade Physics Tournament 2018.

For better understanding, sharing and academic communication.

![](https://github.com/Neuromancer43/Sandpile/blob/master/sandpile_top_view.gif)


Top view of simulated sandpile

![](https://github.com/Neuromancer43/Sandpile/blob/master/figs/time_curve2.png=400px)

Slope angle of sandpile against Time

## Contents
1. automata.py: class of cellular automaton, doctest and comments included 
2. simulation.py: code to run simulation and parameter sweep include u1,u2, k
3. animation.py: visualization, simulation data -> animation, saved to .mp4 files for both top and side views
4. cal_angle: calulate the repose angle of sandpile, generate angle-time curve
5. cal_energy: calulate the total energy and collision loss at every iteration, time curve generated
6. plot.py: plot figures with simulation data

