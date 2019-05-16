# Formula CheatSheet

## Linear Regression Model

## Logistic Regression Model

### Cost Functions

#### Base Cost Function

$$ J(\theta) = \tfrac{1}{m} \sum_{i=1}^{m}Cost(h_{\theta}(x^{(i)},y^{(i)}))$$
$$ Cost(h_{\theta}(x),y) = -log(h_{\theta}(x)) \quad if \ y = 1$$
$$ Cost(h_{\theta}(x),y) = -log(1 - h_{\theta}(x)) \quad if \  y = 0$$

#### Some Properties

$$ Cost(h_{\theta}(x),y) = 0 \quad \  if\  h_{\theta}(x) = y$$
$$ Cost(h_{\theta}(x),y) \rightarrow \infty \quad if\ y = 0 \  and \  h_{\theta}(x) \rightarrow 1$$
$$ Cost(h_{\theta}(x),y) \rightarrow \infty \quad if\ y = 1 \  and \  h_{\theta}(x) \rightarrow 0$$

#### Simplified Cost Function

$$ J(\theta) = \tfrac{1}{m} \sum_{i=1}^{m}[y^{i} log(h_{\theta}(x^{(i)})) + (1 - y^{(i)}) log(1-h_{\theta}(x^{(i)})) $$

A Vectorized implementation is:

$$ h=g(X\theta) $$
$$ J(\theta) = \tfrac{1}{m} * (-y^{T}log(h)-(1-y)^{T}log(-h)) $$

### Gradient Descent for Logistic Regression

#### A vectorized implementation

$$ \theta: = \theta - \tfrac{a}{m} X^{T}(g(X\theta) - \vec{y}) $$

## Regularized Linear Regression

Regularized Linear Regression is a way to reduce the problem of overfitting by introducing a parameter $\lambda$, the higher you set $\lambda$, the smaller significany of that feature is.

### Gradient Descent

We will modify our gradient descent function to separate out $\theta_{0}$ from the rest of the parameters because we do not want to penalize $\theta_{0}$ from the rest of the parameters because we do not want to penalize $\theta_{0}$.

Repeat {  

$$ \theta_{0} := \theta_{0} - a * \tfrac{1}{m} \sum_{i=1}^{m} (h_{\theta}(x^{(i)}) - y^{(i)}))x_{0}^{(i)} $$
$$ \theta_{j} := \theta_{j} - a * [ (\tfrac{1}{m}) \sum_{i=1}^{m} (h_{\theta}(x^{(i)}) - y^{(i)}))x_{j}^{(i)}] + \tfrac{\lambda}{m} \theta_{j} \ \ \ \ for\ j\ in\  {1,2...n}$$

}

#### Simplified version

$$ \theta_{j} := \theta_{j}(1 - a\tfrac{\lambda}{m} -  a (\tfrac{1}{m}) \sum_{i=1}^{m} (h_{\theta}(x^{(i)}) - y^{(i)})x_{j}^{(i)}$$

### Normal Equation

$$ \theta = (X^{T}X + \lambda L)^{-1} X^{T}y $$

where L is identity matrix with $L_{(0,0)}$ is 0