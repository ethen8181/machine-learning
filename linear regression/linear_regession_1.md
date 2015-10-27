# A RoadTrip with Linear Regression
Ming-Yu Liu  

> To all of the machine-learning experts out there, I'm sure people who are not in the statistics, math or computer science department will be very grateful if you could provide simple code examples to go along with mathematical notations.

The following tutorial assumes you know the basics of R programming, e.g. data structure such as list and data.frame, looping, visualization using ggplot2, simple usage of the dplyr package etc. So we will not go over line by line of the code, but will tell split the code into moderate size chunks and tell you what the code is doing.

## Background Information

For supervised learning problems like linear regression, the way it works is when given some set of numbers input variable, we wish to predict another set of numbers. For instance given the number of bedrooms and the size of the house, we wish to predict the price in which the house will be sold. So what we want to know, is how much do "variables" such as the number of bedrooms or the size of the house affects the house's price. One easy approach to calculate these "variables" is by gradient descent.

## Getting Started With Gradient Descent 

Let's start from a basic formula $1.2\times(x-2)^2 + 3.2$. If you still remember basic calculus, the first derivative of function gives you the optimal solution to that funciton. In this case, it's by solving $2\times1.2\times(x-2)=0$.


```r
# libraries that we'll use throughout the tutorial
library(grid)
library(ggplot2)
suppressMessages(library(dplyr))

# original formula 
Formula <- function(x) 1.2 * (x-2)^2 + 3.2

# visualize the function, and the optimal solution
ggplot( data.frame( x = c( 0, 4 ) ), aes( x ) ) + 
stat_function( fun = Formula ) + 
geom_point( data = data.frame( x = 2, y = Formula(2) ), aes( x, y ), 
	        color = "blue", size = 3 ) + 
ggtitle( expression( 1.2 * (x-2)^2 + 3.2 ) )
```

![](linear_regession_1_files/figure-html/unnamed-chunk-1-1.png) 

By solving the first derivative of the function or simply eyeballing the graph, we can easily tell that the minimum value to the formula is when x equals 2. Plugging the value back into the formula gives us the solution 3.2. 

That was easy, however if the function starts to get really complicated then solving this won't be this simple. This is where "gradient descent" comes in, and the formula to this buzzword is listed below.

$$\text{Repeat until converge} \{ x:=x-\alpha\bigtriangledown F(x) \}$$

- The notation := stands for overwriting the value on the left of := with values on the right of := .
- $\bigtriangledown$ stands for taking the first derivative of the function.
- $\alpha$ stands for the learning rate which is set manually.

Let's break that down piece by piece. Putting the formula in plain English: Imagine gradient descent as when you're at the top of a mountain and you want to get down to the very bottom, you have to choose two things. First the direction you wish to descend and second the size of the steps you wish to take. After choosing both of these things, you will keep on taking that step size and that direction until you reach the bottom. Now back to the formula. $\alpha$ corresponds to the size of the steps you wish to take and $\bigtriangledown F(x)$ gives you the direction that you should take for your given formula. The following code, is a small example starting with one iteration of the process. Note that in order for the formula to start calculating you will have to assign an initial value for x.


```r
# first derivative of the formula above
Derivative <- function(x) 2 * 1.2 * (x-2) 

# define the alpha value (learning rate)
learning_rate <- .6

# define the initial value 
x <- .1

( iteration <- data.frame( x = x, value = Formula(x) ) )
```

```
##     x value
## 1 0.1 7.532
```

Here, we defined the learning rate to be 0.6 for our algorithm and the initial value of x equals 0.1 leads to the value 7.532, which is still far off from the optimal solution 3.2. Let's use these user-defined initial value and learning rate on our gradient descent algorithm.


```r
#### One iteration :
# apply the formula of gradient descent
x <- x- learning_rate * Derivative(x)

# output
rbind( iteration, c( x, Formula(x) ) )
```

```
##       x    value
## 1 0.100 7.532000
## 2 2.836 4.038675
```

Row one of the output denotes the intial x and the formula's value when plugging in x, and the second row denotes the value after the first iteraion. We can see that the gradient descent algorithm starts tuning x with the goal of finding smaller value for formula. 

Hopefully, after that one iteration example, everything is a now bit more clear. Before we apply the whole algorithm, I still owe you the definition of "repeat until converge" from the algorithm. 

There usually two ways of applying the notion "repeat until converge" into code. One : Keep updating the x value until the difference between this iteration and the last one, is smaller than epsilon (we use epsilon to denote a user-defined small value). Two : The process of updating the x value surpass a user-define iteration. Often times, you can use both in your first trial, since you probably have no idea when will the algorithm converge, and this is what you'll be seeing in the following code.

Now we're ready to apply the whole thing. 


```r
#### Gradient Descent implementation :

## Define parameters :
# x_new : initial guess for the x value
# x_old : assign a random value to start the first iteration 
x_new <- .1 
x_old <- 0

# define the alpha value (learning rate)
learning_rate <- .6

# define the epilson value, maximum iteration allowed 
epsilon <- .05
step <- 1
iteration <- 10

# records the x and y value for visualization ; add the inital guess 
xtrace <- list() ; ytrace <- list()
xtrace[[1]] <- x_new ; ytrace[[1]] <- Formula(x_new)
cbind( xtrace, ytrace )
```

```
##      xtrace ytrace
## [1,] 0.1    7.532
```

Elaboration on what i defined :

- `x_old` and `x_new` to calculate the difference of the x value between two iterations, the two numbers are different so that the loop can still work for the first iteration of the while loop, as you'll see later.
- `epsilon` value to specify if the difference between `x_old` and `x_new` is smaller than this value then the algorithm will halt.
- `iteration` The maximum iteration to train the algorithm. That is, if the difference of the x value on the 10th iteration and 10 still larger than the `epsilon` value, the algorithm will still halt.
- `xtrace` and `ytrace` stores the x and its corresponding formula value for each iteration. It's good to store these values to get a sense of how fast the algorithm converges.


```r
while( abs( x_new - x_old ) > epsilon & step <= iteration )
{
	# update iteration count 
	step <- step + 1	
	
	# gradient descent
	x_old <- x_new
	x_new <- x_old - learning_rate * Derivative(x_old)
	
	# record keeping 
	xtrace[[step]] <- x_new
	ytrace[[step]] <- Formula(x_new)	
}

# create the data points' dataframe
record <- data.frame( x = do.call( rbind, xtrace ), y = do.call( rbind, ytrace ) )
record
```

```
##          x        y
## 1 0.100000 7.532000
## 2 2.836000 4.038675
## 3 1.632160 3.362368
## 4 2.161850 3.231434
## 5 1.928786 3.206086
## 6 2.031334 3.201178
## 7 1.986213 3.200228
```

From the output above, we can see that the algorithm converges at iteration 2 before the user-specified maximum iteration, with the x and formula value that's close to the original optimal value. Let's create the visualization for the process.


```r
# create the segment between each points (gradient steps)
segment <- data.frame( x = double(), y = double(), xend = double(), yend = double() )
for( i in 1:( nrow(record)-1 ) )
{
	segment[ i, ] <- cbind( record[ i, ], record[ i+1, ] )	
}

# visualize the gradient descent's value 
ggplot( data.frame( x = c( 0, 4 ) ), aes( x ) ) + 
stat_function( fun = Formula ) + 
ggtitle( expression( 1.2 * (x-2)^2 + 3.2 ) ) + 
geom_point( data = record, aes( x, y ), color = "red", size = 3, alpha = .8, shape = 2 ) +
geom_segment( data = segment , aes( x = x, y = y, xend = xend, yend = yend ), 
              color = "blue", alpha = .8, arrow = arrow( length = unit( 0.25, "cm" ) ) )
```

![](linear_regession_1_files/figure-html/unnamed-chunk-6-1.png) 

The visualization gives us a clearer picture that after assigning an inital value of x and parameters such as `epsilon`, `learning_rate`, `iteration`, the gradient descent algorithm will start manipulate the x value until it meets the converge criteria, and an interesting feature is that when the algorithms starts to get closer and closer to the optimal value, it will take smaller "steps " and wander nearby before converging.

Some takeways for this section :

1. The paramters' value you choose will most likely affect your result. Especially for `learning_rate`, for this parameter, if you chosen a value that is too big, the algorithm may skip the optimal value and if you chosen a value that is too small, then the algorithm may take too long to converge. You can try it out yourself ~ .
2. The formula $1.2\times(x-2)^2 + 3.2$ is the "cost function" for this problem, that is we plug in value x into this function to determine whether or not we can reached the optimum. We'll be using this name in the following tutorial, so make sure you have it in your mind.

## Applying it to Linear Regression

Now that we've gotten ourselves familiar with gradient descent, it's now time to apply the same concept to linear regression. We'll use the a simple housing data to illustrate the idea.


```r
housingdata <- read.csv("housing.csv")
list( head(housingdata), dim(housingdata) )
```

```
## [[1]]
##    price area bedrooms
## 1 399900 2104        3
## 2 329900 1600        3
## 3 369000 2400        3
## 4 232000 1416        2
## 5 539900 3000        4
## 6 299900 1985        4
## 
## [[2]]
## [1] 47  3
```

This is a dataset that contains 3 columns (or so called variables) including the number of bedrooms, the area (size) of the house, and 47 rows. For this example, what linear regression will try to do is to train a model using this dataset and in the future after only obtaining the area and bedroom number we want to be able to predict the prices of other houses. So how can we use gradient descent to formulate this linear regression model? Recall that the algorithm's formula is :
$$\text{Repeat until converge} \{ x:=x-\alpha\bigtriangledown F(x) \}$$

Now all we have to do is to define the appropriate cost function of the linear regression and plug it back in to formula above! Before we give it to you, we'll first denote some simple math notations.

- m : the number of training examples. Here we just use all 47 rows.
- n : the number of "input" variables, in this case, it is 2, the number of bedrooms and the area (size) of the house.
- $x_{i}$ : the ith row of the "input" variable in the dataset.
- $y_{i}$ : the ith row "output" variables. In this case the output variable is the price of the house.
- Formula for linear regression :

$$ h_{\theta}(x) = \theta_{0} + \theta_{1}x_{1} + \theta_{2}x_{2} + \dotsm + \theta_{n}x_{n}  $$

Here the $\theta_{j}$s denotes the weights or so called coeffficients of the model for each varible. Here we only have 2 variables, so j only goes up to 2, and the h(x) for this dataset would be $\theta_{0} + \theta_{1}x_{area} + \theta_{2}x_{bedrooms}$. And these $\theta_{j}$s are what we want find out or train. So given a training dataset, how do we learn the parameters $\theta_{j}$?

One reasonable method is to make the value produced by function F(x) to be as close to the original $y_{i}$ as possible (at least for the training dataset we now have). That is after plugging in the combinations of the number of bedrooms and the area size of the house into the function, we want the house price calculated by the function to be as close to the original value of the house price as possible (for every row of dataset). This gives us the cost function $F(\theta)$ below.

$$ F(\theta) = \frac{1}{2} \sum_{i=1}^m ( h_{\theta}(x_{i}) - y_{i} )^2 $$

The $\frac{1}{2}$ is there to minimize the math loading for later when we take the first derivative of the function. Again, the meaning for the formula above means after plugging in our input variables $x_{i}$ (recall that i denotes the ith row in the dataset) into the function and obtaining the value, which is the $h(x_{i})$ part. We will calculate its distance with the original $y_{i}$. Therefore the process of the gradient descent is to start some value for $\theta$ and keep updating it to reduce $F_{\theta}$, with the goal of minimizing the summed up differences for all rows. This summed of difference is often referred to as the sum squared error. 

Next, we'll state without proving that after taking the first derivative of the function $F_{\theta}$ and putting it back inside the gradient descent algorithm we'll obtain the formula below:

In progress, work on it later.

## References

1. Gradient Descent Example: http://www.r-bloggers.com/gradient-descent-in-r/
2. Linear Regression with Gradient Descent: http://cs229.stanford.edu/notes/cs229-notes1.pdf 
