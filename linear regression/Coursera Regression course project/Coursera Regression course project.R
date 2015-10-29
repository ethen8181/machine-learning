# You work for Motor Trend, a magazine about the automobile industry. 
# Looking at a data set of a collection of cars, they are interested in exploring the 
# relationship between a set of variables and miles per gallon (MPG) (outcome). 
# They are particularly interested in the following two questions:
# Is an automatic or manual transmission better for MPG
# "Quantify the MPG difference between automatic and manual transmissions"

library(ggplot2)

# mpg : miles/ gallon
# am, Transmission (0 = automatic, 1 = manual )
# ggplot( mtcars, aes( factor(am), mpg, fill = factor(am) ) ) + geom_dotplot( binaxis = "y", stackdir = "center" )  
mtcars$am <- factor( mtcars$am, levels = c( 0, 1 ), labels = c( "Automatic", "Manual") )
ggplot( mtcars, aes( factor(am), mpg, fill = factor(am) ) ) + geom_boxplot()

tapply( mtcars$mpg, mtcars$am, mean )
# data <- split( mtcars$mpg, mtcars$am )
# ks.test( data[[1]], data[[2]] )
# ks.test( data[[2]], "pnorm" )

# a test comparing the variances between the two is applied prior to conducting the t-test
var.test( mpg ~ am, data = mtcars, alternative = "two.sided" )
# From the p value, we can infer that there is insufficient evidence that the variance differ
# that is, they are the same between the two measurements.

t.test( mpg ~ am, paired = FALSE, var.equal = TRUE, data = mtcars )
# mean between the two is significantly different

# getting the confidence interval
# attach(mtcars)
# mpgaut <- mpg[am == 0]
# mpgman <- mpg[am == 1]
# x1 <- t.test(mpgaut, mu = 0)
# x2 <- t.test(mpgman, mu = 0)
# x1$conf.int


model1 <- lm( mpg ~ am, data = mtcars )
summary(model1)$coef

list( coefficient = summary(model1)$coef, rsquare = summary(model1)$r.squared )



col <- which( names(mtcars) %in% c( "cyl", "vs", "gear", "carb" ) )

data <- modifyList( mtcars, lapply( mtcars[,col], as.factor ) )
sapply(mtcars,class)
sapply(data,class)

line <- lm( mpg ~ ., data = mtcars )

line <- update( line, .~.-cyl )

line <- update( line, .~.-vs )

line <- update( line, .~.-carb )

line <- update( line, .~.-gear )

line <- update( line, .~.-drat )

line <- update( line, .~.-disp )

line <- update( line, .~.-hp )
summary(line)

# final formula for the regression line
model2 <- lm( formula = mpg ~ wt + qsec + am, data = data )

summary(model2)
# wt Weight (lb/1000)
# qsec 1/4 mile time

# plotting the predicted and actual value 
# there should be strong correlation between the model¡¦s predictions and its actual results
data1 <- data.frame( actual = mtcars$mpg, fitted = model2$fitted.values )
ggplot( data1, aes( fitted, actual ) ) + geom_point() + geom_smooth( method = "lm" )

# plotting the predicted value and the residuals
# Ideally your plot of the residuals should be 
# (1) they're pretty symmetrically distributed, tending to cluster towards the middle of the plot
# (2) they're clustered around the lower single digits of the y-axis (e.g., 0.5 or 1.5, not 30 or 150)
# (3) in general there aren¡¦t clear patterns
data2 <- data.frame( residuals = model2$residuals, fitted = model2$fitted.values )
ggplot( data2, aes( fitted, residuals ) ) + geom_point() + geom_hline( yintercept = 0, color = "blue", size = 1 )







