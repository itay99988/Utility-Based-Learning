from utility_based import *
from arguments import get_args
from examples import example_selector


def repeat_experiment(args):
    avg_time, avg_samples_count, success_count = 0, 0, 0
    is_dual_mode = args.mode.lower() == 'dual'

    for _ in range(args.repeat):
        dfa, params = example_selector(args.experiment, is_dual_mode, False)
        con_dfa, acc_prefixes, rej_prefixes, time_elapsed = single_run(dfa=dfa, params=params, Dual=is_dual_mode)

        if con_dfa.is_pos_neg_consistent(acc_prefixes, rej_prefixes) and len(dfa.states) == len(con_dfa.states):
            print("Success")
            avg_time += time_elapsed
            avg_samples_count += (len(acc_prefixes) + len(rej_prefixes))
            success_count += 1
        else:
            print("Failed")

    print("\n\nResults for utility based experiment: {}, {} repetitions".format(dfa.name, args.repeat))
    print("Success rate: {}%".format((success_count/args.repeat)*100))
    print("Average number of required samples: {}".format(avg_samples_count/success_count))
    print("Average time: {} s".format(avg_time/success_count))


if __name__ == '__main__':
    args = get_args()  # Parse arguments from command line
    repeat_experiment(args)
