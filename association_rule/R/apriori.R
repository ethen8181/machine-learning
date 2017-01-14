library(DT) # for interactive data frame
library(arules)
library(data.table)
wdpath <- normalizePath('/Users/ethen/machine-learning/association_rule/R')
setwd(wdpath)

load('titanic.raw.rdata')
dt <- data.table(titanic.raw)
titanic <- as(dt, 'transactions')
summary( itemFrequency(titanic) )

# train apriori
rules <- apriori( 
    titanic,

	# the min/max len denotes the min/max number of items in a itemset
	parameter = list(support = 0.05, confidence = 0.7, minlen = 2, maxlen = 5),
    
    # for appearance we can specify we only want rules with rhs 
    # containing "Survived" only (we then specfiy the default parameter
    # to 'lhs' to tell the algorithm that every other variables that
    # has not been specified can go in the left hand side
    appearance = list( rhs = c('Survived=No', 'Survived=Yes'), default = 'lhs' ),

	# don't print the algorthm's training message
	control = list(verbose = FALSE)
)


# converting rules' info, such as left and right hand side, and all the quality measures,
# including support, confidence and lift a to data.frame
# http://stackoverflow.com/questions/25730000/converting-object-of-class-rules-to-data-frame-in-r
rules_dt <- data.table( lhs = labels( lhs(rules) ), 
                        rhs = labels( rhs(rules) ), 
                        quality(rules) )[ order(-lift), ]

# -------------------------------------------------------------------------
# not included

# a scatter plot using support and confidence on the x and y axes. 
# and the lift is used as the color of the points
library(cowplot)
library(ggplot2)

ggplot( rules_dt, aes(support, confidence, color = lift) ) +
geom_point() + 
labs( title = sprintf( 'scatter plot for %d rules', nrow(rules_dt) ) )


# confirm that the toy python code's result matches R's apriori
X = matrix(c(1, 1, 0, 0, 0, 0,
             1, 0, 1, 1, 1, 0,
             0, 1, 1, 1, 0, 1,
             1, 1, 1, 1, 0, 0,
             1, 1, 1, 0, 0, 1), ncol = 6, byrow = TRUE)

rules <- apriori( 
    X,
    
    # the min/max len denotes the min/max number of items in a itemset
    parameter = list( support = 0.5, confidence = 0.5, minlen = 2, maxlen = 5 ),
    
    # don't print the algorthm's training message
    control = list( verbose = FALSE )
)

