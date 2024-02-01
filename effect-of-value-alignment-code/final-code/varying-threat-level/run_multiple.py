import argparse
import _context
from Utils import add_common_args
from NonAdaptive import NonAdaptiveRobot
import time


def main():
    parser = argparse.ArgumentParser()
    parser = add_common_args(parser)

    # Add specific arguments for this script
    parser.add_argument('--d-start', type=float, help='Threat level to start the simulation (default:0.0)', default=0.0)
    parser.add_argument('--d-end', type=float, help='Threat level to start the simulation (default:1.0)', default=1.0)
    parser.add_argument('--grid-step', type=float, help='stepsize on the weights grid (default:0.01)', default=0.05)

    args = parser.parse_args()
    stepsize = args.grid_step
    d_start = args.d_start
    d_end = args.d_end

    num_levels = int((d_end - d_start) / stepsize) + 1

    d_list = [d_start + stepsize * i for i in range(num_levels)]
    data_directory = "./data/OneStepOptimal/rob{:1.1f}/hum{:1.1f}/".format(args.health_weight_robot,
                                                                              args.health_weight_human)

    for d in d_list:
        timestamp = time.strftime("%Y%m%d-%H%M%S") + '/'
        args.threat_level = d
        nar = NonAdaptiveRobot(args)
        nar.run(data_directory, timestamp)


if __name__ == "__main__":
    main()
