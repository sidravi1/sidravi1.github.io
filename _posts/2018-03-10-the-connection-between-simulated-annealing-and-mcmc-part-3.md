---
layout: "post"
title: "The connection between Simulated Annealing and MCMC (Part 3)"
date: "2018-03-10 15:00"
use_math: true
comments: true
---

Check out [part 1]({{ site.baseurl }}{% post_url 2018-03-01-the-connection-between-simulated-annealing-and-mcmc-part-1 %}) and [part 2]({{ site.baseurl }}{% post_url 2018-03-02-the-connection-between-simulated-annealing-and-mcmc-part-2 %}). Let's start off by writing the code for the Metropolis algorithm and comparing it to Simulated Annealing.

{% highlight python %}
def metropolis(p, qdraw, nsamp, stepsize, initials):

   samples=np.empty((nsamp, 5))
   x_prev =initials['xinit']

   for i in range(nsamp):

       x_star = qdraw(x_prev, stepsize)
       p_star = p(x_star)
       p_prev = p(x_prev)

       # note that p is in logs!
       pdfratio = p_star - p_prev                  # (1)
       threshold = np.random.uniform()
       alpha = min(1, np.exp(pdfratio))

       if threshold < alpha:
           samples[i] = x_star                     # (2)
           x_prev = x_star

       else:
           samples[i]= x_prev                      # (3)

    return samples

{% endhighlight %}


Here's the simulated annealing algorithm again but using the same variable names where it makes sense.

{% highlight python %}
def annealing(p, qdraw, nsamp, initials):

    L = initials['L']
    T = initials['T']

    x_prev =initials['xinit']
    for e in range(nsamp):
        for l in range(L):
            x_star = qdraw(X)                              
            p_star = p(x_star)
            p_prev = p(x_prev)   

            pdfratio = p_prev - p_star              # (1)
            threshold = np.random.uniform()                
            alpha = min(1, np.exp(pdfratio/T))      

            if (p_star < p_prev) or (threshold < alpha ):
                x_prev = x_star

               # Let's keep track of the best one
               if p_star < p_prev:
                   x_best = x_star                  # (2)

        # Let's calculate new L and T
        T = T * 0.85                                  
        L = L * 1.1                                   

    return x_best
{% endhighlight %}

Let's start off by writing the code for the Metropolis algorithm and comparing it to Simulated Annealing.

## Similarities and differences

1. In both, we have a proposal function that generates an *x_star*
2. In both, we calculate the probability of the target distribution at *x_star*
3. Temperature for Metropolis is just 1 (T = 1).
4. Note that in SA, we move from *x_prev* to *x_star* in SA if the new energy, or the value of the target function, is lower. In Metropolis, we move if the probability mass function of the target distribution is higher.
5. We always take a sample in Metropolis.

It's pretty remarkable how similar they are.

## Why does it work?

The main reason this works is because of "detailed balance" which you may remember from the post on Markov Chains. We said that if $\pi_{i} p_{ij} = p_{ij} \pi_{j}$ then $\pi$ is a stationary distribution. Detailed balance for continuous variables is:

$$

f(x)p(x,y) = f(y)p(y,x)

$$

Now, if this is true (and proof isn't too hard), then you know that *f* is a stationary distribution for the chain. And since it is stationary, each step in the Markov Chain lands us back in *f*. This means that all the samples that we draw are coming from *f*. 

If you squint hard enough, you'll notice that even in simulated annealing, we're drawing from a stationary distribution at each temperature.

It may take a little while for the chain to get to that stationary distribution depending on where you start but once you're there you're (by definition) stuck in the stationary distribution. But how do you know that you actually made it to your stationary distribution? The truth is that you don't really. There are signs that you *haven't* made it to the stationary distribution so high dose of paranoia is advised.

We'll talk about *burnin*, *autocorrelation*, Gewecke, and Gelman-Rubin test, in a future post.

{% if page.comments %}
  {%- include disqus_tags.html -%}
{% endif %}
