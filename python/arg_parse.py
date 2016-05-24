# argparse 

# https://docs.python.org/3/howto/argparse.html

import argparse

# Basics

"""
parser = argparse.ArgumentParser()
parser.add_argument( "echo", help = "echo the string you use here" )
args = parser.parse_args()
print args.echo
"""

# 1. add_argument : 
# 	 specify which command-line options the program is willing to accept

# 2. Calling our program now requires us to specify the additional argument
# 	 the parse_args() function now returns some data and the data's name
# 	 matches the string argument given to the method

# 3. adding help to make us know what it does 


# -----------------------------------------------------------
# Specifying the type 

"""
parser = argparse.ArgumentParser()
parser.add_argument( "square", 
					 help = "display a square of a given number",
					 type = int )
args = parser.parse_args()
print args.square ** 2
"""

# 1. argparse treats the options we give it as strings, so 
# 	 we have to specify the type 


# -----------------------------------------------------------
# Optional arguments 

parser = argparse.ArgumentParser()
parser.add_argument( "-v", "--verbose", 
					 help = "increase output verbosity",
					 action = "store_true" )
args = parser.parse_args()
if args.verbose:
    print "verbosity turned on"

# 1. --verbose option is actually optional, 
# 	 there is no error when running the program without it

# 2. -v is the short option for it 

# 3. action = "store_true" means that, if the option is specified, 
# 	 assign the value True to args.verbose. Not specifying it implies False. 
# 	 it will complain when you specify the value that's not boolean














