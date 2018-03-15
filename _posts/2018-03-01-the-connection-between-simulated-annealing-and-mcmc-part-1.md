---
layout: post
title: The connection between Simulated Annealing and MCMC (Part 1)
date: '2018-03-01 15:23'
use_math: true
comments: true
---

I was going to dive straight into it but thought I should go over Simulated Annealing (SA) first before connecting them. SA is an heuristic optimization algorithm to find the global minimum of some complex function $f(X)$ which may have a bunch of local ones. Note that $X$ can be vector of length N: $X = [x_1, x_2, ..., x_n]$

SA is pretty straight forward. The following is some pseudo-code (though it's similarity to python is uncanny):

```python
# Initialize:

X_old =       # propose some X in the support of f(X)
E_old = f(X)  # E is for energy
T = 100       # pick some large value of T
L = 10        # this is the length of the first epoch
```


- X_old is the starting point so pick whatever makes sense or come up with one randomly.
- T should be large enough. You'll probably end up tweaking this as you go.
- L can be small. It will grow exponentially as T decreases.

```python
# Stopping condition
Epoch = 1000
```

- There are a number of ways you could do this (e.g. When T reaches a certain level or when E is not being updated). We'll keep it simple here.

```python
# Let's anneal!
for e in range(Epoch):
    for l in range(L):
         X_new = propose(X)                        # (1)
         E_new = f(X_new)

         threshold = U(0,1)                        # (2)
         alpha = min(1, e^((E_old - E_new)/T)      # (3)

         if (E_new < E_old) or (alpha > threshold):
              E_old = E_new
              X_old = X_new

              # Let's keep track of the best one
              if E_new < E_old:
                   X_best = X_new

     # Let's calculate new L and T
     T = T * 0.85                                  # (4)
     L = L * 1.1   
```

That's it! Pretty disappointing right?

### (1) How do you propose a new X

There are some technical requirements:

- It's based on the old X: It should be in the neighbourhood of X_old BUT you should be able to get to any value in the X space from any other value using this proposal i.e. the entire space of X that we are interested in should be "communicable". In Markov language, the proposal function must be irreducible.
- It should be symmetric: You should be able to get back to X_old from X_new with equal likelihood or the same proposal function. The fancy name for this is "detailed balance".

That's all you need. But practically, you don't want your proposal to be too narrow and look for things that are too similar else you'll just be stuck around your local minima. You also don't want it to be too wild where you propose something way radical, else you hop around a lot more than you'd like and will have a low acceptance rate. Finding that balance is an art. Or do as I do and try a few different proposals.

### (2) Threshold

Just a random number drawn from a uniform distribution.

### (3) alpha

This only kicks in if $ E_{old}$ is less than $ E_{new}$. And therefore $ e^{(E_{old} - E_{new}) / T}$ is a small number. The min makes sure that alpha stays below (or equal to) 1.

So if the solution (X) you are suggesting has higher energy (E or f(X)), then accept it sometimes. 'Sometimes' is based on how much higher energy this new solution requires, what temperature (T) you're at and some good-old-fashioned randomness.

### (4) Reducing T

We want to gradually reduce the temperature. To ensure convergence, we should really do $ T(t) = \frac{T_0}{ln(1 + t)}$ for $t = 1, 2, ...$ but this takes forever. So something like what we have done is fine.

### (5) Increasing L

At the start you are bouncing around since T is high and you accept a lot of stuff. Later on, you'll be accepting fewer things so you should spend a lot of time proposing solutions and slowly descending towards your minima. So L is small to start off with but large once the system has cooled.

In [Part 2]({{ site.baseurl }}{% post_url 2018-03-02-the-connection-between-simulated-annealing-and-mcmc-part-2 %}), let's explore this a little more and then move on to connecting this with MCMC.

{% if page.comments %}
  {%- include disqus_tags.html -%}
{% endif %}
