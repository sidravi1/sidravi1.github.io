---
layout: "post"
title: "Poisson Density Estimation with Gaussian Processes"
date: "2018-08-22 09:27"
comments: true
use_math: true
---

I have been slowly working my way through Efron & Hastie's [Computer Age Statistical Inference (CASI)](https://web.stanford.edu/~hastie/CASI_files/PDF/casi.pdf). Electronic copy is freely available and so far it has been a great though at time I get lost in the derivations.

In chapter 8, E&H show an example of a Poisson density estimation from a spatially truncated sample. Here, I implement the example using pymc3 and in addition to the latent linear model, I explore Gaussian Processes for the latent variable.

You can get the notebook [from here](https://github.com/sidravi1/Blog/blob/master/nbs/GP_Poisson_density_estimation.ipynb).

## The Problem

You can download the data from [here]("https://web.stanford.edu/~hastie/CASI_files/DATA/galaxy.txt"). From CASI:

> ... shows galaxy counts from a small portion of the sky: 486 galaxies have had their redshift $r$ and apparent magnitude $m$ measured. Distance from earth is an increasing function of $r$, while apparent brightness is a decreasing function of $m$.

Note that this is truncated data. $r$ and $m$ are limited to

$$
1.22 \le r \le 3.32\\
17.2 \le m \le 21.5
$$

Here's what the data table like:

![Raw Data]({{"/assets/2018_08_22_data.png" | absolute_url}})

Our job is to fit a density to this truncated data. We are going to model this as

$$
y_i \sim Poisson(\mu_i),\qquad i = 1,2,...,N
$$

So each observation comes from a Poisson with a mean $\mu_i$. Actually, what we'll model instead is $\lambda_i = \log(\mu_i)$.

## Using Linear latent variable

The way Efron & Hastie model $\lambda$ is:

$$
\mathbf{\lambda}(\alpha)= \mathbf{X}\,\alpha
$$

Where $\mathbf{X}$ is a "structure matrix" made up for features we crafted:

$$
\mathbf{X} = [\mathbf{r, m, r^2, rm, m^2}]
$$

Where $\mathbf{r^2}$ is the vector whose components are the square of $\textbf{r}$'s etc. If we think that the true density is bivariate normal, then this makes sense. The log density of a bivariate normal is of this form.

Let's setup this model in pymc3:

{% highlight python %}
with pm.Model() as basic_model:

    alpha_0 = pm.Normal('alpha_0', 0, 10)
    alpha_1 = pm.Normal('alpha_1', 0, 10)
    alpha_2 = pm.Normal('alpha_2', 0, 10)
    alpha_3 = pm.Normal('alpha_3', 0, 10)
    alpha_4 = pm.Normal('alpha_4', 0, 10)
    alpha_5 = pm.Normal('alpha_5', 0, 10)

    latent = pm.Deterministic('latent', tt.dot(data_mat_n, tt.stack([alpha_0, alpha_1, alpha_2, alpha_3, alpha_4, alpha_5])))

    likelihood = pm.Poisson('like', tt.exp(latent), observed=data_long['val'].values)
{% endhighlight %}

Let's see what our density looks like:

![Linear Model]({{"/assets/2018_08_22_linear_model.png" | absolute_url}})

Because we setup a Bayesian GLM, not only do we have the mean values for $\mu$, we have the full distribution. The plot on the right shows the the standard deviation and as you might have expected, it's high in the region where we didn't have any data.


## Using a Gaussian Process

If we are comfortable being agnostic of the functional form -- and since I am not an astrophysicist, I'm quite comfortable -- we can use a 2-d gaussian process. You may want to out the previous posts on Gaussian Processes:

1. [Intro to gaussian processes]({{ site.baseurl }}{% post_url 2018-04-03-gaussian-processes %})
2. [Latent GP and binomial likelihood]({{ site.baseurl }}{% post_url 2018-05-15-latent-gp-and-binomial-likelihood %})

So now our lambda is:

$$
\mathbf{\lambda_i} = f^* (x_i)
$$

Here $x$ is $(r, m)$, a point on the grid and $f(x)$ is a Gaussian Process:

$$
f(x) \sim GP(m(x), k(x, x'))\\
f^* (x) = f(x) | y
$$

(Sorry about the loose notation).

Here's the model in pymc3:

{% highlight python %}
with pm.Model() as gp_model:

    ro = pm.Exponential('ro', 1)
    eta = pm.Exponential('eta', 1)

    K = eta**2 + pm.gp.cov.ExpQuad(2, ro)
    gp = pm.gp.Latent(cov_func=K)

    # Note that we are just using the first two columns of
    # data_mat which correspond to 'r' and 'm'
    latent = gp.prior('latent', X=data_mat[:,:2])

    likelihood = pm.Poisson('like', tt.exp(latent), observed=data_long['val'].values)

{% endhighlight %}

And the resulting density:

![Linear Model]({{"/assets/2018_08_22_gp_model.png" | absolute_url}})

The standard deviation surface looks a lot bumpier but the general trend is similar; it is higher in areas where we have little data. Very cool, but is that model any good?

## Deviance residual

Efron & Hastie use the deviance residual to check the goodness of fit.

$$
Z = sign(y - \hat{y})D(y, \hat{y})^{1/2}
$$

Side note: this is based on KL-divergence so I couldn't figure out how they got around the problem of 0. For KL-divergence you need to have non-zero probabilities in the entire domain. I just ignore the 'NaNs' resulting from it and pretend like nothing happened.

{% highlight python %}
def deviance_resid(y, mu):
    def D(mu1, mu2):
        return  2 * mu1 * (((mu2/mu1) - 1) - np.log(mu2/mu1))

    return np.sign(y - mu) * np.sqrt(D(y, mu))
{% endhighlight %}

Comparing the sum of squares of the residual, we get 109.7 for the GP model and 134.1 for the linear model. This next plot shows these residuals.

<div id="vis3"></div>

<script type="text/javascript">
  var spec = "{{"/assets/2018_08_22_comparing_resid.json" | absolute_url}}";
  var opt = {"actions":false}
  vegaEmbed('#vis3', spec, opt).then(function(result) {
    // access view as result.view
  }).catch(console.error);
</script>

Looks like the GP model fits the top left corner better than the linear model but it also fits that stray bottom right dot that makes me worry that it might be overfitting. One way around it would be to use a different implementation of GP that allows for additive noise.

We can check for overfitting by looking at WAIC for the two using pymc3's `compareplot`:

![Linear Model]({{"/assets/2018_08_22_compare_plot.png" | absolute_url}})

Looks like the GP-model is the better model. Check out [the notebook ](https://github.com/sidravi1/Blog/blob/master/nbs/GP_Poisson_density_estimation.ipynb) and let me know if there are ways you would improve this. How can we implement this using `gp.marginal` or `gp.conditional` so that we can include additive noise?

{% if page.comments %}
  {%- include disqus_tags.html -%}
{% endif %}
