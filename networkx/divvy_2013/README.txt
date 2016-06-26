This file contains metadata for both the Trips and Stations table.

For more information, see the contest page at http://DivvyBikes.com/datachallenge or email questions to data@DivvyBikes.com. 


Metadata for Trips Table:

Variables:

trip_id: ID attached to each trip taken
starttime: day and time trip started, in CST
stoptime: day and time trip ended, in CST
bikeid: ID attached to each bike
tripduration: time of trip in seconds 
from_station_name: name of station where trip originated
to_station_name: name of station where trip terminated 
from_station_id: ID of station where trip originated
to_station_id: ID of station where trip terminated
usertype: "Customer" is a rider who purchased a 24-Hour Pass; "Subscriber" is a rider who purchased an Annual Membership
gender: gender of rider
birthyear: birth year of rider


Notes:

* First row contains column names
* Total records = 759,789
* Trips that did not include a start or end date were removed from original table.
* Gender and birthday are only available for Subscribers



Metadata for Stations table:

Variables:

name: station name    
latitude: station latitude
longitude: station longitude
dpcapacity: number of total docks at each station as of 2/7/2014
online date: date the station went live in the system
