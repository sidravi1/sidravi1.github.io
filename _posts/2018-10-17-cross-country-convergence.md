---
layout: "post"
title: "You'll be blown a way by this weird trick millennials discovered to do convergence regressions."
date: "2018-10-17 00:07"
comments: true
use_math: true
---

Hat tip to [@mkessler_DC](https://twitter.com/mkessler_DC/status/1051959149494448128) for the clickbaitey title.

So economic convergence is back. Dev Patel, Justin Sandefur, and Arvind Subramanian (PSS) recently wrote a [blog post](https://www.cgdev.org/blog/everything-you-know-about-cross-country-convergence-now-wrong) to show that
> while unconditional convergence was singularly absent in the past, there has been unconditional convergence, beginning (weakly) around 1990 and emphatically for the last two decades.

They do a very straightforward analysis; they regress real per capita GDP growth rate on the log of per capita GDP for various periods and show that the coefficient is indeed reducing. They also made their code available, which is so awesome that they can be forgiven for it being in STATA. Instead of doing separate regressions for each time period, I redo their analysis as a rolling regression where we allows the coefficients to change over time.

This post borrows heavily from Thomas Weicki's [two](https://docs.pymc.io/notebooks/GLM-rolling-regression.html) [posts](https://twiecki.github.io/blog/2017/03/14/random-walk-deep-net/) on Gaussian random walks.

As per usual, you can find the [notebook on github](https://github.com/sidravi1/Blog/tree/master/nbs/growth_analysis).

## Why rolling regression?

Key figure in the chart is this:

![Raw Data](https://www.cgdev.org/sites/default/files/patel-sandefur-subramanian-beta_by_series.png)

So for each time period between $y$ and $y_0$, we have different coefficients as follows:

$$
\Delta GDPPC_{c, y, y_0} = \alpha_{y, y_0} + \beta_{y, y_0} * log(GDPPC_{c, y0})
$$

Going forward we will drop $y_0$ from the notation for clarity.

This implicit assumption here is that $\beta_{y}$ is independent from  $\beta_{y + 1}$. But we know that's not entirely true. If you gave me $\beta_{y}$, I'd expect $\beta_{y + 1}$ to be pretty similar and deviate only slightly.

We can make this explicit and put a gaussian random-walk prior on beta.

$$
\begin{align}
\beta_{y + 1} \sim \mathcal{N}(\beta_{y}, \sigma^2) \tag{1}
\end{align}
$$

$\sigma^2$ is how quickly $\beta$ changes. It's just a hyper-parameter that we'll put a weak prior on. We do the same thing for $\alpha$ as well.

## Pre-processing

In the notebook, I tried replicate PSS's STATA code in python as closely as possible. It's not perfect. And looks like I screwed up the WDI dataset manipulations so I use only Penn World Table (PWT) and Maddison Tables (MT) in this post. Check out the notebook if you want to fix it and get it working for WDI.

Here's what 1985 looked like:

<div id="vis_1985"></div>

<script type="text/javascript">
  var spec = "{{"/assets/2018_10_17_growth_1985.json" | absolute_url}}";
  var opt = {"actions":false}
  vegaEmbed('#vis_1985', spec, opt).then(function(result) {
    // access view as result.view
  }).catch(console.error);
</script>

And here's what 2005 looked like:

<div id="vis_2005"></div>

<script type="text/javascript">
  var spec = "{{"/assets/2018_10_17_growth_2005.json" | absolute_url}}";
  var opt = {"actions":false}
  vegaEmbed('#vis_2005', spec, opt).then(function(result) {
    // access view as result.view
  }).catch(console.error);
</script>

The trend might not be so obvious in 1985 but it does appear to be negative in 2005, especially if you ignore those really tiny countries. Let's see what the random walk model says.

##  Setup in pymc3

Here's the model in pymc3 for MT. The code for PWT is pretty identical.

{% highlight python %}
with pm.Model() as model_mad:

    sig_alpha = pm.Exponential('sig_alpha', 50)
    sig_beta = pm.Exponential('sig_beta', 50)

    alpha = pm.GaussianRandomWalk('alpha', sd =sig_alpha, shape=nyears)
    beta = pm.GaussianRandomWalk('beta', sd =sig_beta, shape=nyears)

    pred_growth = alpha[year_idx] + beta[year_idx] * inital
    sd = pm.HalfNormal('sd', sd=1.0, shape = nyears)

    likelihood = pm.Normal('outcome',
                           mu=pred_growth,
                           sd=sd[year_idx],
                           observed=outcome)
{% endhighlight %}

- `nyears` is the number of years. So there are `nyears` alphas and betas and they are *linked* as per (1).
- `year_idx` indexed to right coefficients based on $y$.
- As in PSS, `outcome` is $\Delta GDPPC_{c, y, y_0}$ and `intial` is $log(GDPPC_{c, y0})$.

and let's run it:

{% highlight python %}
trace_mad = pm.sample(tune=1000, model=model_mad, samples = 300)
{% endhighlight %}

##  Results

Let's see how alpha and beta change over the years.

![Linear Model]({{"/assets/2018_10_17_coefficient_drift_pwt_mad.png" | absolute_url}})

So, PSS's conclusion is borne out here as well. We see that $beta$ used to be mildly positive and gradually shifted to negative in recent years. Our bounds look a little tighter and maybe the shape is slightly different. You can check out [the notebook](https://github.com/sidravi1/Blog/tree/master/nbs/growth_analysis) here. I threw this together pretty quickly so I might be screwed up the data setup. Let me know if you find any errors in my code.


{% if page.comments %}
  {%- include disqus_tags.html -%}
{% endif %}
