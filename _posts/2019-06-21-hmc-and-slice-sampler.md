---
layout: "post"
title: "Back to Basics with David Mackay #4: HMC and Slice sampler - now with animations!"
date: "2019-06-21 20:34"
comments: true
use_math: true
---

I just wanted to put up a few animations of HMC and slice samplers that I have been playing around with.

I started coding up the slice sampler and the Hamiltonian Monte-Carlo (HMC) after reading the chapter in MacKay's book. Around the same time, Colin Carol published is `minimc` package for HMC. This package so nicely written so I just decided to play around with it instead. I'll do another post with some HMC experiments.

The code for the HMC animation (uses minimc) can be [found here](https://gist.github.com/sidravi1/a7965d57c63e71f9b9ff47098cd774df). The code for slice sampler and the file to run the animation can be [found here](https://github.com/sidravi1/slicesampler).

## Hamlitonian Monte Carlo

I would recommend heading over to Colin's blog and reading the introduction to HMC he has posted up there and then come back and watch these videos. Note the Metropolis proposals where the sampler jumps in momentum space to a new energy level. Then it takes a bunch of leapfrog steps and then takes a sample.

{% include youtube.html id="po0Obe04Nfs" %}

## 1-d slice sampler

I basically implemented Mackay's algorithm from his book (Ch 29, page 375):

**Main algorithm**
![slice sampler main]({{"/assets/20190622_slice1.png" | absolute_url}})

**Stepping out**
![slice sampler step out]({{"/assets/20190622_slice2.png" | absolute_url}})

**Shrinking**
![slice sampler shrinkage]({{"/assets/20190622_slice3.png" | absolute_url}})

And here's the video:

{% include youtube.html id="NsYqQt0Yfs8" %}

## multi-dimensional sampler

Toward the end of his lecture on slice sampling, MacKay answers a student question about how you'd do a multi-dimensional slice sampling. That makes a hell of lot more sense to me that his one paragraph in the book. Basically, each time we pick a random direction for the vector we step out in. Rest is pretty much the same.

{% include youtube.html id="Ti3DRyUDWZU" %}

{% if page.comments %}
  {%- include disqus_tags.html -%}
{% endif %}
