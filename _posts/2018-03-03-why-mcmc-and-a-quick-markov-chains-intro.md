---
layout: "post"
title: "Why MCMC and a quick markov chains intro"
date: "2018-03-03 22:10"
use_math: True
comments: true
---
A lot of this material is from Larry Wasserman's All of Statistics. I love how the title makes such a bold claim and then quickly hedges by adding the subtitle "A *Concise* Course in Statistical Inference" (The italic are mine).

Before systematically trying to understand MCMC, I floundered a bit. [This post](https://jeremykun.com/2015/04/06/markov-chain-monte-carlo-without-all-the-bullshit/) by Jeremy Kun is probably one of the best explanations but that too required some side reading and close following with a scratch pad. I'm going to try to explain it simply (glossing over some details and proofs) and slowly, but if it's too basic, you might want to head over to Jeremy's site (you should be reading his blog anyway). This post start off with Markov Chain - "All of Markov Chains: A Concise Intro to Markov Chains" and then we'll continue onto Metropolis, Metropolis-Hastings, and all the other good stuff.

## Storytime

This "story" about Good King Markov is from section 8.1 of "Statistical Rethinking" by Richard McElreath. King Markov's kingdom has 10 islands {i<sub>1</sub>... i<sub>10</sub>}, each island is neighboured by 2 others so they all form a ring. They are all of different sizes and so have different populations. Population of island i<sub>1</sub> is P<sub>1</sub>. Population of island i<sub>2</sub> is P<sub>2</sub> = 2P<sub>1</sub>, and Population of island i<sub>3</sub> is P<sub>3</sub> = 3P<sub>1</sub> and so forth.

The king wants to spend time on the islands in proportion to their population size. King Markov, like your humble writer, doesn't like planning out his schedule in advance. And, because he's lazy, like your humble writer, wants to only travel to neighbouring islands. How should he do it?

Here's how he does it:

1. Start of each week, he flips a coin. If it's heads he considers moving clockwise, if it's tails he considers moving anti-clockwise.
2. To decide whether he actually moves, he gets # of shells equivalent to the island number that has been proposed by the coin. So if it is i<sub>6</sub>, gets 6 shells out. Then gets out the number of stones equivalent to the island number he is on. So if he is on i<sub>7</sub>, he gets out 7 stones and 5 if he is on i<sub>5</sub>.
3. If there are more shells than stones, move to the island.
4. If there are fewer shells than stones, discard the number of stones equal to the number of shells. So in our example, he has 7 stones so he discards 6 of them.  He puts the 1 stone remaining and the 6 shells back into a bag and gives it a vigourous shake.
5. He reaches in and pull out one object. If it is a shell, he moves, else he stays for another week. So probability of staying in 1/7 and moving is 6/7.

This works! The king's movement looks erratic but in the long run, it satisfies his requirement and allows him to remain a monarch in an increasingly democratizing world.

This is a special case of the Metropolis algorithm (which is one example of MCMC algorithm). Let that stew for a bit or even better try and simulate this in python or R to convince yourself that this works.

## Why MCMC?

From Jeremy Kun's post that I mentioned earlier but the word in bold is mine:

> Markov Chain Monte Carlo is a technique to solve the problem of **efficiently** sampling from a complicated distribution. 

He does a great job of motivating it so read that if you have more "why"s. In brief, you have some distribution *f(X)*, and you want to draw i.i.d samples such that their distribution is *f(X)*. There are less efficient ways of doing it (see rejection sampling) but if you're talking about high dimensional space, you'll need this.

## Markov chains

A Markov chain is a stochastic (read:random) process where the distribution of X<sub>t</sub> depends only on X<sub>t-1</sub>, where (X<sub>1</sub>, ... , X<sub>t</sub>) are random variables. So it has memory of just 1 time period.

It's given by a transition matrix, *P*,  with element *p<sub>ij</sub>* that are the probability of going from state i to j.  One more notation: p<sub>ij(n)</sub> is the probability of going from i to j in n steps.

### Some interesting things about Markov chains

For all of these, I talk about the properties of the state. But if all states have a property then the chain itself can be said to have this properties.

I'm skipping the proofs but they are pretty straight forward:

- $p_{ij}(m + n) = \sum_k p_{ik}(m)p_{kj}(n)$ : To get the probability of getting from i to j in (m+n) steps we are summing over all the possible paths where we get to some intermediate stage, k, in m steps and then from there to j in n steps.
- P<sub>n</sub> = P<sup>n</sup>: To calculate the probabilities of state transition in n-steps is just P<sub>n</sup>.

### Some properties of Markov chains

For all of these, I talk about the properties of the state. But if all states have a property then the chain itself can be said to have this properties.

- **Communicable**: If you can get from i to j in some n with a non-zero probability then i and j are communicable. If all states can communicate with each other, it is irreducible.
- **Recurrent**: Say you are in state i. State i is recurrent if you will return to i if you keep applying the transition matrix. Else it is transient. Cool fact: A finite Markov chain must have at least one recurrent state. Also, if it is irreducible, then all states are recurrent.
- **Aperiodic**: State i is aperiodic if it isn't a cycle. That is when you will be in in a state i should be not be determined solely by n.
Non-null: If mean time taken to get back to state i is less than $\infty $ it is non-null.
- **Ergodic**: If a state is recurrent, non-null, aperiodic, it is ergodic.
- **Stationarity**: If say $\pi$ is a stationary distribution if $\pi = \pi \cdot P$
- **Limiting Distribution**: $\pi_j = \lim_{n \to \infty} P^n_{ij}$ independant of i. So no matter which state we start in, if we run it for long enough, the probability of being in state j is $\pi_j$
- **Detailed balance**: $\pi_{i} p_{ij} = p_{ij} \pi_{j}$ for all i,j.  This takes a minute to get your head around. It saying something very strong: if there is a distribution where, for all the states, the amount going in from another state is equal to the amount going back to that state, then that distribution is a stationary distribution.

Ok. Next time, we'll get back to going from simulated annealing to MCMC.

{% if page.comments %}
  {%- include disqus_tags.html -%}
{% endif %}
