import itertools
import platform
from pathlib import Path
from automata.fa.dfa import DFA
from RLUtility import RLUtility, CommonUtils
from examples import *
from sklearn.cluster import DBSCAN
from collections import defaultdict
import numpy as np
import time


# remove unused states - as a final step of automaton construction
def remove_unused_states(init_state, built_states, built_transitions):
    tr_to_remove = []
    st_to_remove = []
    for s in built_states:
        if s != init_state:
            unused = True
            for t in built_transitions:
                if t.target_state == s:
                    unused = False
            # if s is unused, remove it from built transitions
            if unused:
                st_to_remove.append(s)
                for t in built_transitions:
                    if t.source_state == s:
                        tr_to_remove.append(t)

    for t in tr_to_remove:
        built_transitions.remove(t)
    for s in st_to_remove:
        built_states.remove(s)


# used as part of automaton construction phase
def find_transition_target(action, prefixes, S2C, state_len):
    cluster_counter = [0]*len(set(S2C.values()))

    for p in prefixes:
        if len(p) < state_len and p+action in S2C:
            target_cluster = S2C[p+action]
            cluster_counter[int(target_cluster[1:])] += 1

    max_val = max(cluster_counter)

    return 'q' + str(cluster_counter.index(max_val))


# compatibility test for the dual clustering approach
def compatible_test_clustering(p1, p2, actions, S2C1, S2C2, state_len, acceptance_dict_tst):
    if S2C2[p1] == -1 or S2C2[p2] == -1:
        return True

    if S2C2[p1] != S2C2[p2]:
        if acceptance_dict_tst[p1] == acceptance_dict_tst[p2]:
            print(f"err for comparing({p1},{p2})")
        return False

    # successors
    for act in actions:
        if not compatible_test_clustering(p1 + act, p2 + act, actions, S2C1, S2C2, state_len, acceptance_dict_tst):
            return False

    return True


# compatibility test for the utility based with lookahead approach
def compatible_test_rec(p1, p2, actions, acceptance_dict):
    if acceptance_dict[p1] == -1 or acceptance_dict[p2] == -1:
        return True

    if acceptance_dict[p1] != acceptance_dict[p2]:
        return False

    for act in actions:
        if not compatible_test_rec(p1 + act, p2 + act, actions, acceptance_dict):
            return False

    return True


# this function represents the main algorithm, with both lookahead and dual clustering approaches
def single_run(dfa: Process, params: GeneralParameters, Dual: bool = False):
    # blackbox automaton
    original_dfa = CommonUtils.process2dfa(dfa)

    # create a log directory for the desired experiment and visualize the blackbox automaton
    Path(f"logs/{dfa.name}/{platform.node()}/{params.num_iterations}").mkdir(parents=True, exist_ok=True)
    original_dfa.show_diagram(path=f"logs/{dfa.name}/{platform.node()}/original.png")
    file = open(f"logs/{dfa.name}/{platform.node()}/{params.num_iterations}/prefix_tree_learning.dat", 'w+')

    actions = list(params.reward_dict.keys())

    # Monte Carlo policy evaluation
    start_monte_carlo = time.time()
    monte_carlo_p, samples_n = RLUtility.monte_carlo(dfa=dfa, params=params)
    end_monte_carlo = time.time()

    # first clustering
    start_dbscan = time.time()

    values_to_process = np.array(list(monte_carlo_p.V1.values())).reshape(-1, 1)
    clustering_res = DBSCAN(eps=params.eps, min_samples=params.min_samples).fit(values_to_process)
    S2C = defaultdict(lambda: -1)
    for i in range(len(monte_carlo_p.states1)):
        S2C[monte_carlo_p.states1[i]] = "s" + str(clustering_res.labels_[i])
    C2S = RLUtility.invert_S2C(S2C)

    initial_clusters = list(set(S2C.values()))

    if Dual:
        # second, course-grained clustering for the dual clustering approach
        values_to_process = np.array(list(monte_carlo_p.V2.values())).reshape(-1, 1)
        clustering_res = DBSCAN(eps=params.eps2, min_samples=params.min_samples).fit(values_to_process)
        S2C2 = defaultdict(lambda: -1)
        for i in range(len(monte_carlo_p.states2)):
            S2C2[monte_carlo_p.states2[i]] = "s" + str(clustering_res.labels_[i])

        comp_clusters = list(set(S2C2.values()))

        print("comp clusters: ", comp_clusters)
        print("clusters after RL: ", initial_clusters)

    end_dbscan = time.time()

    ############################################### End of RL Phase ###############################################

    # Set up acceptance data structures
    start_structure_build = time.time()

    acc_prefixes = list(monte_carlo_p.accepting_prefixes)
    rej_prefixes = list(monte_carlo_p.rejecting_prefixes)

    acceptance_dict = defaultdict(lambda: -1)
    for p in acc_prefixes:
        acceptance_dict[p] = True
    for p in rej_prefixes:
        acceptance_dict[p] = False

    end_structure_build = time.time()

    # Splitting the initial clusters - with accept/reject information or with coarse clustering information
    start_prefix_tree = time.time()
    new_clusters_count = 0
    for cluster in initial_clusters:
        buckets = []
        prefixes_in_cluster = C2S[cluster]

        if prefixes_in_cluster:
            buckets.append([])
            buckets[0].append(prefixes_in_cluster[0])
            for prefix_idx in range(1, len(prefixes_in_cluster)):
                bucket_idx = 0

                # compatible test for dual clustering approach
                if Dual:
                    while bucket_idx < len(buckets) and (not compatible_test_clustering(p1=prefixes_in_cluster[prefix_idx],
                                                                                        p2=buckets[bucket_idx][0],
                                                                                        actions=actions,
                                                                                        S2C1=S2C,
                                                                                        S2C2=S2C2,
                                                                                        state_len=params.STATE_LEN,
                                                                                        acceptance_dict_tst=acceptance_dict)):
                        bucket_idx += 1
                # compatible test for utility based with lookahead approach
                else:
                    while bucket_idx < len(buckets) and (not compatible_test_rec(p1=prefixes_in_cluster[prefix_idx],
                                                                                    p2=buckets[bucket_idx][0],
                                                                                    actions=actions,
                                                                                    acceptance_dict=acceptance_dict)):
                        bucket_idx += 1

                if bucket_idx >= len(buckets):
                    buckets.append([])
                buckets[bucket_idx].append(prefixes_in_cluster[prefix_idx])

            # re-assign clusters to each prefix based on the splitting information
            for bucket in buckets:
                for prefix in bucket:
                    S2C[prefix] = "q" + str(new_clusters_count)

                new_clusters_count += 1

    ######################################### End of Splitting Phase ############################################

    # Clusters were re-assigned. Now we can move on to automaton construction.
    # This step is identical for both approaches
    C2S = RLUtility.invert_S2C(S2C)

    # Inferring states
    built_states = list(set(S2C.values()))

    # Inferring the initial state.
    built_init_state = S2C['']

    # Inferring the accepting states
    built_accepting_states = set()
    for p in monte_carlo_p.states1:
        if p in monte_carlo_p.accepting_prefixes:
            built_accepting_states.add(S2C[p])

    # Inferring transitions
    built_transitions = []
    is_consistent = True
    for built_state in built_states:
        for action in actions:
            target_built_state = find_transition_target(action, C2S[built_state], S2C, params.STATE_LEN)
            built_transitions.append(Transition(action, built_state, target_built_state, 0))

    # remove unused states
    remove_unused_states(built_init_state, built_states, built_transitions)

    # Initializing the constructed dfa object.
    con_dfa = Process('constructed_dfa', states=built_states, transitions=built_transitions,
                      initial_state=built_init_state, accepting_states=built_accepting_states)
    end_prefix_tree = time.time()

    print("final clusters: ", built_states)

    ######################################### End of Construction Phase ############################################

    # visualize the learned automaton
    result_dfa = CommonUtils.process2dfa(con_dfa)
    result_dfa.show_diagram(path=f"logs/{dfa.name}/{platform.node()}/{params.num_iterations}/{dfa.name}.png")

    time_elapsed = end_monte_carlo-start_monte_carlo + \
                   end_dbscan - start_dbscan + \
                   end_structure_build - start_structure_build + \
                   end_prefix_tree-start_prefix_tree

    print(f"{dfa.name} utility based learning - {time_elapsed} s")
    print(f"{dfa.name} utility based learning - {time_elapsed} s", file=file)

    return con_dfa, acc_prefixes, rej_prefixes, time_elapsed


if __name__ == '__main__':
    Dual = False
    RPNI = False
    dfa, params = cross7(Dual=Dual, RPNI=RPNI)
    con_dfa, acc_prefixes, rej_prefixes, time_elapsed = single_run(dfa=dfa, params=params, Dual=Dual)

    # comparison between original and constructed dfa
    identical_dfa = con_dfa.is_pos_neg_consistent(acc_prefixes, rej_prefixes) and len(dfa.states) == len(con_dfa.states)
    print("Success: ", identical_dfa) # True means equal
