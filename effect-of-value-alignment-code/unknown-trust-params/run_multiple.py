import argparse
import _context
from classes.Utils import add_common_args
from NonAdaptive import NonAdaptiveRobot
#from Adaptive import AdaptiveRobot
#from APriorAlignment import APrioriAlignedRobot
import time

def main():
    parser = argparse.ArgumentParser()
    parser = add_common_args(parser)
    
    # Add specific arguments for this script
    # parser.add_argument('--health-weight-robot', type=float, help='Fixed health weight of the robot (default: 0.7)', default=0.7)
    # parser.add_argument('--health-weight-human', type=float, help='True health weight of the human (default: 0.9)', default=0.9)
    parser.add_argument('--wh-start', type=float, help='health weight to start the grid (default:0.0)', default=0.0)
    parser.add_argument('--wh-end', type=float, help='health weight to end the grid (default:1.0)', default=1.0)
    parser.add_argument('--grid-step', type=float, help='stepsize on the weights grid (default:0.05)', default=0.05)
    # parser.add_argument('--threat-level', type=float, help='The prior threat level across all houses (default=0.5)', default=0.5) # Added in common args
    
    args = parser.parse_args()
    stepsize = args.grid_step
    wh_start = args.wh_start
    wh_end = args.wh_end

    num_weights = int((wh_end-wh_start)/stepsize) + 1
    
    wh_list = [wh_start + stepsize * i for i in range(num_weights)]
    wh_list_all = [stepsize * i for i in range(int(1/stepsize) + 1)]
    data_direc_base = "./data/ThreatLevel/{:1.1f}/wh_start_{:1.1f}/".format(args.threat_level, args.wh_start)

    for wh_rob in wh_list_all:
        args.health_weight_robot = wh_rob
        for wh_hum in wh_list:
            data_direc = data_direc_base + time.strftime("%Y%m%d-%H%M%S") + '/'
            args.health_weight_human = wh_hum
            nar = NonAdaptiveRobot(args)
            nar.run(data_direc)

if __name__ == "__main__":
    main()

    