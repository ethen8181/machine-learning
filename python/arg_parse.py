# walking through the argparse tutorial
# https://docs.python.org/3/howto/argparse.html

import argparse

# -----------------------------------------------------------
# example 1

"""
parser = argparse.ArgumentParser()
parser.add_argument( "square", help = "display a square of a given number", type = int )
args = parser.parse_args()
print(args.square ** 2 )
"""

"""
1. add_argument with .add_argument ,
specify which command-line options the program is willing to accept

2. Calling our program now requires us to specify the additional argument,
the parse_args() function now returns some data and the data's name
matches the string argument given to the method

3. adding help to make us know what it does 

4. specify the type, by default it treats the input as a string
"""

# -----------------------------------------------------------
# example 2

"""
parser = argparse.ArgumentParser()
parser.add_argument( "square", help = "display a square of a given number", type = int )
parser.add_argument( "-v", "--verbose", 
					 help = "increase output verbosity",
					 action = "store_true" )
args = parser.parse_args()
answer = args.square ** 2
if args.verbose:
	print("the square of {} equals {}".format( args.square, answer ))
else:
	print(answer)
"""

"""
1. --verbose option is actually optional, 
there is no error when running the program without it

2. -v is the short option for it 

3. action = "store_true" means that, if the option is specified, 
assign the value True to args.verbose. Not specifying it implies False. 
it will complain when you specify the value
"""

parser = argparse.ArgumentParser()
parser.add_argument( "--num1", help = "number1", required = True, type = int )
parser.add_argument( "--num2", help = "number2", required = True, type = int )
args = parser.parse_args()
arguments = vars(args) # use vars to convert all the parameters into a dictionary
print(arguments)



