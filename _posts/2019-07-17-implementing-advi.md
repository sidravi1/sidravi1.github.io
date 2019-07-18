---
layout: "post"
title: "Implementing ADVI with autograd"
date: "2019-07-17 22:17"
comments: true
use_math: true
---

We use things without knowing how they work. Last time my fridge stopped working, I turned it off and on again to see if that fixed it. When it didn't I promptly called the "fridge guy". If you don't know how things work, you don't know when and how they break, and you definitely don't know how to fix it.

I've been interviewing a ton of people. It's hard to find the "ML guy" who can fix a broken model but there is no dearth of people who know how to train some ML model.

Rant over. On with the show.

## Implementing ADVI

Avid readers may remember that I'm working my way through Mackay's lectures and book. He has a lecture or two on variational inference, where he approximates posterior states of the Ising model. He handcrafts $q(\theta)$ for the model and finds the parameters that minimize KL-divergence from $p(\theta \vert x)$.

With Automatic Differentiation Variational Inference (ADVI), you don't need to worry about all that work. You have probably read the paper - it's 3 years old now which is ancient history in the CS world - but [here it is again](https://arxiv.org/pdf/1603.00788.pdf) for reference.

## The algorithm

We're going to fit a simple linear regression model.

$$
Y \sim N(X'\beta, 2.5)
$$

We're going to choose $\beta$ to be [5, -3.5]. We'll generate $y$ and see if we can recover these betas.

The algorithm itself is brilliant in its simplicity.

### 1. Transform latent variables ###

If support is constrained (e.g. is just positive) then we transform it to have a support over all real-values. Why do we need this? Remember that we are trying to minimize KL-divergence:

$$
\phi^* = \text{arg min}\\
\DeclareMathOperator*{\argmax}{argmax}
\argmax_{\phi \in \Phi} \text{KL}(q(\theta; \phi) \Vert p(\theta | x))
$$

and KL divergence is:

$$
KL(q\Vert p) = \int _ {-\infty }^{\infty} q(x) \log \frac{q(x)}{p(x)}\,dx
$$

Now if you choose an $x$ such that $p(x)$ is zero, we're going to have a division by zero and a blackhole will appear under your chair. So we need the following condition to be true:

$$
\text{supp}(q(\theta;\phi)) \subseteq \text{supp}(p(\theta\vert x))
$$

In order to do our inference automatically, we need to transform $p(\theta \vert x)$ to have a support over all reals. Then we can just choose a $q(\theta)$ that has a support over all reals as well... like your friendly neighbourhood gaussian.

One transformation might be $\zeta = \log(\theta)$. Note that this takes a $\theta$ with a support of positive reals and transforms it to $\zeta$, which has a support over all reals.

### 2. Calculate gradients

Another transformation allows you to convert the gradient of the objective function as an expectation over the gradients of the joint density: $\nabla_{\theta}\log p(x, \theta)$. Check out the paper. Basically it's an affine transformation (you're normalizing) so you don't need to stress about Jacobians.

The key takeaway is that if you can calculate the jacobians of the transformation we did in (1) and the gradient of $p(x, \theta)$ then we can calculate the gradient of the ELBO. Enter `autograd`.

#### Autograd

Part of my motivation for this implementation was to play with the `autograd` package. It's so beautifully easy. If you can numpy and scipy, you can autograd... mostly. You can check out the [github readme](https://github.com/HIPS/autograd) on what is implemented.

So you write your functions like you would in numpy/scipy and use the `grad`, `jacobian` and other functions to get the gradient function. Here's an example:

{% highlight python %}
from autograd import numpy as npa

grad_cos = grad(npa.cos)
grad_cos(np.pi / 2)

## -1.0
{% endhighlight %}

In our case, we just need to write out:

$$p(x, \theta) = p(x | \theta)p(\theta)$$

The first bit on the RHS is your likelihood and the second bit is the prior. Both super easy to code up. If you have multiple thetas, you can either assume they are independent (mean-field) or also fit their covariance (full-rank). For our linear model with mean-field it is simply:

{% highlight python %}

# Prior
def log_p_theta(self, betas, sigma):
    beta_prior = spa.stats.norm.logpdf(betas, self.betas_mu, self.betas_sd).sum()
    sigma_prior = spa.stats.gamma.logpdf(sigma/self.sigma_scale,
                                         self.sigma_shape) - npa.log(self.sigma_scale)

    return beta_prior + sigma_prior

# Likelihood
def log_p_x_theta(self, theta):
    # likelihood
    betas = theta[:2]
    sigma = theta[2]
    ones = np.ones((self.x.shape[0],1))
    x = np.hstack([ones, self.x])
    yhat = (x @ betas)
    like = spa.stats.norm.logpdf(self.y, yhat, sigma).sum()

    return like + log_p_theta(betas, sigma)

{% endhighlight %}

and the gradients can be gotten easily. Here we code up equations (8) and (10) from the paper.

{% highlight python %}
def nabla_mu(self, eta):

    x = self.x
    y = self.y

    zeta = (eta * self.omega) + self.mu
    theta = self.inv_T(zeta)

    grad_joint = elementwise_grad(self.log_p_x_theta)(theta)
    grad_transform = elementwise_grad(self.inv_T)(zeta)
    grad_log_det = elementwise_grad(self.log_det_jac)(zeta)
    return grad_joint * grad_transform + grad_log_det

def nabla_omega(self, nabla_mu_val, eta):
    return nabla_mu_val * eta.T * npa.exp(self.omega) + 1

def log_det_jac(self, zeta):
    a = jacobian(inv_T)(zeta)
    b = npa.linalg.det(a)
    return npa.log(b)

{% endhighlight %}

### 3. Stochastic optimization ###

We take a draw from a standard gaussian, do all the inverse-transform, calculate gradients and then just do gradient ascent. You can even mini-batch this to make it even faster but we don't do it here.

The authors also provide their own algorithm to adaptively set the step-size that combines RMSPROP with a long memory. Here it is in python:

{% highlight python %}
def get_learing_rate(base_lr, iter_id, s, gk, tau = 1, alpha=0.9):
    s = alpha * gk**2 + (1 - alpha) * s
    rho = base_lr * (iter_id ** (-1/2 + 1e-16)) / (tau + s**(1/2))

    return rho, s
{% endhighlight %}

Where `s` is initialized as `s = gk**2`.

## Show me the results

Here's an animation of the optimization process. Enjoy!

{% include youtube.html id="DyRjj3ufuAw" %}

## Code and last words

You can get [the code here](https://github.com/sidravi1/Blog/tree/master/src/advi). If you really want the animation code, then ping me and I'll clean up the notebook and throw it up somewhere.

Full-rank doesn't look too hard to do. That would be a nice extension if you want to do it. I did play around with other transformations, especially the $\log(\exp(\theta) - 1)$ and at least for this example, I didn't find much of a difference. If you do that, don't forget to use `logaddexp` when writing the inverse transform - else you will get underflow/overflow.

Choosing the base learning rate took a little tuning. Also, my biggest time sink was not realizing that you shouldn't transform (well you can't take the log of a negative!) the latent params that already have a support over all reals. Oof.
