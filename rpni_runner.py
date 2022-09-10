import random
from automata.fa.dfa import DFA
from PTA import PTA
from RLUtility import RLUtility, CommonUtils
from RPNI import RPNI
from arguments import get_args
from examples import *
import time


def random_validator_pta(result_dfa: PTA, original_dfa: DFA):
    for _ in range(1000):
        random_str = ''.join(random.choice(list(original_dfa.input_symbols)) for i in
                             range(random.randint(0, len(original_dfa.states) * 5)))

        if original_dfa.accepts_input(random_str) != result_dfa.get_output_4_prefix(random_str)[1]:
            return False

    return True


def random_validator(result_dfa: DFA, original_dfa: DFA):
    if len(result_dfa.states) != len(original_dfa.states):
        return False

    for _ in range(1000):
        random_str = ''.join(random.choice(list(original_dfa.input_symbols)) for _ in
                             range(random.randint(0, len(original_dfa.states) * 5)))

        if original_dfa.accepts_input(random_str) != result_dfa.accepts_input(random_str):
            return False

    return True


def learn_4_parameters_prefix(original_process: Process, original_dfa: DFA, num_iterations_param: int, episode_len_param: int):
    start_time = time.time()

    # build pta
    pta, pos_examples, neg_examples = CommonUtils.build_pta_from_blackbox(blackboxProc=original_process,
                                                                          blackbox=original_dfa,
                                                                          episode_len=episode_len_param,
                                                                          num_iterations=num_iterations_param)

    learner = RPNI(neg_examples=neg_examples, alphabet=original_dfa.input_symbols)
    result = learner.learn(blackbox=original_dfa, initial_pta=pta)
    total_samples = len(neg_examples) + len(pta.pos_states)
    end_time = time.time()

    try:
        result_dfa = CommonUtils.pta2dfa(result)
    except Exception as e:
        print(e)
        if random_validator_pta(result, original_dfa):
            return original_dfa, total_samples, end_time - start_time
        else:
            return None, None, None

    # validate
    d1 = result_dfa.difference(original_dfa)
    d2 = original_dfa.difference(result_dfa)

    # random validator
    isEq = d1.isempty() and d2.isempty() and random_validator(result_dfa, original_dfa)

    return isEq, total_samples, end_time - start_time


def repeat_experiment(args):
    avg_time, avg_samples_count, success_count = 0, 0, 0

    for _ in range(args.repeat):
        dfa, params = example_selector(args.experiment, dual=False, RPNI=True)
        original_dfa = CommonUtils.process2dfa(dfa)

        success, total_samples, time_elapsed = learn_4_parameters_prefix(original_process=dfa,
                                                                          original_dfa=original_dfa,
                                                                          num_iterations_param=params.num_iterations,
                                                                          episode_len_param=params.episode_len)
        if success:
            print("success")
            avg_time += time_elapsed
            avg_samples_count += total_samples
            success_count += 1
        else:
            print("failed")

    print("\n\nResults for RPNI experiment: {}, {} repetitions".format(dfa.name, args.repeat))
    print("Success rate: {}%".format((success_count/args.repeat)*100))
    print("Average number of required samples: {}".format(avg_samples_count/success_count))
    print("Average time: {} s".format(avg_time/success_count))


if __name__ == '__main__':
    args = get_args()  # Parse arguments from command line
    repeat_experiment(args)
