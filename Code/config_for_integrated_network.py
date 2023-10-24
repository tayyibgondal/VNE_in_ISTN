num_of_iter = 8 # Number of iterations for cyclic movement simulation
requests = [100,150,200,250] # no of offline requests generated in each run of the simulation
num_points_along_orbit = 140 # no of equally spaced points along elliptical orbit of satellites
communication_range = 90 # arbitrary value (scaled down 10 times from the 1125 km altitude)
no_of_runs = 1 # number of times the simulation is repeated to get average results

# no of ground nodes in topology
start_value_node = 1
end_value_node = 21
	
# no of satellite nodes in topology
start_value_satellite = 1
end_value_satellite = 81

GROUND_EDGES = 30

# initial bandwidths of links
GROUND_CAPACITY = 10000
S2S_AND_G2S_CAPACITY = 5000
	
# bounds on bandwidth requests
REQUIRED_RATE_MINIMUM = 400
REQUIRED_RATE_MAXIMUM  = 4000

LEO_period = 128 # period of orbit in minutes

max_allowed_latency = 20
