---
layout: post
title: Monte Carlo Methods
date: '2018-03-11 10:40'
use_math: true
comments: true
---

I imagine most of you have some idea of Monte Carlo (MC) methods. Here we'll try and quantify   it a little bit.

Here's a common example. How would you be calculate the area of a circle if a gave you a formula for it's circumference - $ 2 \pi r$. Easy, you say. Just integrate with respect to r and you get $ \pi r^2$.

What if you couldn't easily take the integral? Say you're working with a gaussian pdf and want to take the integral. Not so easy (or even possible) to do it the standard analytical way (that's not entirely true, check out Box-Muller. But we'll call that 'non-standard' so that we are not wrong.).  We fall back on numerical methods like the trapezoid rule or Simpson's rule. MC is one additional method. It is computationally less expensive that the other ways when you're working in very high dimensions ( d > 6, is where it starts beating Simpson's rule).

Back to our trivial but oh-so-illuminating circle example. Let's get the area using a MC. We draw a square around it of size 2r x 2r and throw a bunch of darts at it. We calculate the ratio of darts that fall inside the circle to all the darts thrown. This is the ratio of the area of the circle to the area of square, $ 4r^2 $ (which we know to be simply, pi).

Let's formalize this a bit. Say we want to integrate some function h(x). If we draw from any "easy" (like uniform) distribution f(x) :

$$

I = h(x) f(x) dx\\

\hat{I} = \frac{1}{N} \sum_{x_i~ f} h(x_i)

$$

So the process comes down to:

- Draw N samples from a pdf f(x) that has the same support as h(.)
- Calculate h(.) at each of the N samples you drew.
- Take the average of those values

If you're wondering where, the "divide by the integral of f(x) over the support" step is, f(x) is pdf so we know it is one. Easy peasy.

At this stage, it should start looking very familiar to the MCMC rejection/acceptance process. So why do we even need Markov Chains? Let's just take samples from multi-dimensional uniform distribution (or normal if we want), and do the process above.

This brings us on to the curse of dimensionality which we'll leave for another post.

{% if page.comments %}
  {%- include disqus_tags.html -%}
{% endif %}
