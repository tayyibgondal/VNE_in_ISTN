num_of_iter = 8 # Number of iterations for cyclic movement simulation
requests = [20,50,80,110] # no of offline requests generated in each run of the simulation
num_points_along_orbit = 80 # no of equally spaced points along elliptical orbit of satellites
communication_range = 90.5 # arbitrary value (scaled down 10 times from the 1125 km altitude)

# no of ground nodes in topology
start_value_node = 1
end_value_node = 21

# no of satellite nodes in topology
start_value_satellite = 1
end_value_satellite = 21

# initial bandwidths of links
GROUND_CAPACITY = 5000
S2S_AND_G2S_CAPACITY = 1000

# bounds on bandwidth requests
REQUIRED_RATE_MINIMUM = 50
REQUIRED_RATE_MAXIMUM  = 1000

MAX_LATENCY = 20

LEO_period = 128 # period of orbit in minutes

