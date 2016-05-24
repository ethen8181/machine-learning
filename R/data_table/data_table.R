library(knitr)
library(data.table)
setwd("/Users/ethen/programming/R/data_table")

# -------------------------------------------------
# 						joins
# -------------------------------------------------

employees   <- fread("data/employees.csv")
departments <- fread("data/departments.csv")
knitr::kable( employees, caption = "Table Employees" )
knitr::kable( departments, caption = "Table Departments" )


# inner join with merge
merge( employees, departments, by = "Department" )

# inner join with keys
# set the keys of the tables to represent the by:
setkey( employees, Department )
setkey( departments, Department )
# equivalent to setkeyv( departments, "Department" )

# note that you only need to set the keys once
# you can confirm if it works with
key( employees)

# perform the join,
# nomatch 0 means no rows will be returned 
# from not matched rows from the Right table
employees[ departments, nomatch = 0 ]


# left join with merge, notice that the column's ordering is different
merge( employees, departments, by = "Department", all.x = TRUE )
departments[employees]


# right join with merge
merge( employees, departments, by = "Department", all.y = TRUE )
employees[departments]

# we can also do a not matched with !
# returning the rows in the employees table that are not in the
# department table
employees[!departments]


# full outer join
merge( employees, departments, by = "Department", all = TRUE )


# -------------------------------------------------
# 				 tips and tricks
# -------------------------------------------------

# Get the sum of y and z and 
# the number of rows in each group while grouped by x
DT <- data.table( x = sample( 1:2, 7, replace = TRUE ), 
				  y = sample( 1:13,7 ), 
				  z = sample( 1:14,7 ) ) 
DT[ , c( lapply( .SD, sum ), .N ), by = x ]

# set the keys and
# select the matching rows without using `==`.
DT <- data.table( A = letters[ c( 2, 1, 2, 3, 1, 2, 3 ) ],
				  B = c( 5, 4, 1, 9, 8, 8, 6 ), C = 6:12 )
setkey( DT, A, B )

# matches rows where A == "b"
DT["b"]

# matches rows where A == "b" or A == "c"
DT[ c( "b", "c" ), ]

# match both keys
DT[ .( "b", 5 ), ]

# Subset all rows where just the second key column matches
DT[ .( unique(A), 5 ), ]

# Select the first row of the `b` and `c` groups using `mult`.
DT[ c( "b", "c" ), mult = "first" ]
# DT[ c( "b","c" ), mult = "last" ]

# group by each i
DT[ c( "b", "c" ), .SD[ c( 1, .N ), ] ]
DT[ c( "b", "c" ), .SD[ c( 1, .N ), ], by = .EACHI ]

