import argparse
import _context
from Utils import add_common_args
from Adaptive import AdaptiveRobot
import time


def main():
    parser = argparse.ArgumentParser()
    parser = add_common_args(parser)

    # Add specific arguments for this script
    parser.add_argument('--wh-start', type=float, help='health weight to start the grid (default:0.0)', default=0.0)
    parser.add_argument('--wh-end', type=float, help='health weight to end the grid (default:1.0)', default=1.0)
    parser.add_argument('--grid-step', type=float, help='stepsize on the weights grid (default:0.05)', default=0.1)

    args = parser.parse_args()
    stepsize = args.grid_step
    wh_start = args.wh_start
    wh_end = args.wh_end

    num_weights = int((wh_end - wh_start) / stepsize) + 1

    wh_list = [wh_start + stepsize * i for i in range(num_weights)]
    parent_directory = "./data/BoundedRational/Adaptive/{:1.1f}".format(args.threat_level)

    for wh_hum in wh_list:
        timestamp = time.strftime("%Y%m%d-%H%M%S") + '/'
        args.health_weight_human = wh_hum
        nar = AdaptiveRobot(args)
        nar.run(parent_directory, timestamp)


if __name__ == "__main__":
    main()

