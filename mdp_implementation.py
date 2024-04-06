from copy import deepcopy, copy
import numpy as np


def bellman_equation(mdp, U, curr_i, curr_j, gamma):
    actions = ['UP', 'DOWN', 'RIGHT', 'LEFT']
    actions_num = {0: 'UP', 1: 'DOWN', 2: 'RIGHT', 3: 'LEFT'}
    new_V = float('-inf')
    s_reward = float(mdp.board[curr_i][curr_j])
    for a in actions:
        value = 0
        transition_f = mdp.transition_function[a]
        for index in range(4):
            state = mdp.step((curr_i, curr_j), actions_num[index])
            value += transition_f[index] * float(U[state[0]][state[1]])
        new_V = max(new_V, value)
    U_result = new_V * gamma + s_reward
    return U_result


def value_iteration(mdp, U_init, epsilon=10 ** (-3)):
    # TODO:
    # Given the mdp, the initial utility of each state - U_init,
    #   and the upper limit - epsilon.
    # run the value iteration algorithm and
    # return: the U obtained at the end of the algorithms' run.
    #

    # ====== YOUR CODE: ======
    U_curr = U_init
    discount_factor = mdp.gamma  # gamma
    while True:
        U = deepcopy(U_curr)
        lambda_v = 0
        for i in range(mdp.num_row):
            for j in range(mdp.num_col):
                if mdp.board[i][j] == 'WALL':
                    continue
                if (i, j) in mdp.terminal_states:
                    U_curr[i][j] = float(mdp.board[i][j])
                    continue
                U_s = bellman_equation(mdp, U, i, j, discount_factor)
                U_curr[i][j] = U_s  # U'
                lambda_v = max(lambda_v, abs(U_curr[i][j] - U[i][j]))
        if lambda_v < (epsilon * (1 - discount_factor)) / discount_factor:
            return U
            break
    return U


def get_policy(mdp, U):
    # TODO:
    # Given the mdp and the utility of each state - U (which satisfies the Belman equation)
    # return: the policy
    #

    # ====== YOUR CODE: ======
    actions = ['UP', 'DOWN', 'RIGHT', 'LEFT']
    actions_num = {0: 'UP', 1: 'DOWN', 2: 'RIGHT', 3: 'LEFT'}

    curr_value = 0
    max_policy = ''
    # init policy board , empty
    policy_board = []
    for _ in range(mdp.num_row):
        row = []
        for _ in range(mdp.num_col):
            row.append(0)
        policy_board.append(row)
    # FILL THE BOARD WITH OPTIMAL POLICY ACCORDING TO U !!!
    for i in range(mdp.num_row):
        for j in range(mdp.num_col):
            max_value = float('-inf')
            if mdp.board[i][j] == 'WALL' or (i, j) in mdp.terminal_states:
                continue
            for a in actions:
                value = 0
                transition_f = mdp.transition_function[a]
                for index in range(4):
                    step = mdp.step((i, j), actions_num[index])
                    U_step = float(U[step[0]][step[1]])
                    value += transition_f[index] * U_step
                if value > max_value:
                    max_value = value
                    max_policy = a
            policy_board[i][j] = max_policy
    return policy_board


def policy_evaluation(mdp, policy):
    # TODO:
    # Given the mdp, and a policy
    # return: the utility U(s) of each state s
    #

    # ====== YOUR CODE: ======
    actions_num = {0: 'UP', 1: 'DOWN', 2: 'RIGHT', 3: 'LEFT'}
    gamma = mdp.gamma
    # init utility board , empty
    utility_board = []
    epsilon = 10 ** (-3)
    discount_factor = mdp.gamma
    for _ in range(mdp.num_row):
        row = []
        for _ in range(mdp.num_col):
            row.append(0)
        utility_board.append(row)
    # FILL THE BOARD WITH utility
    while True:
        U = deepcopy(utility_board)
        lambda_v = 0
        for i in range(mdp.num_row):
            for j in range(mdp.num_col):
                curr_sum = 0
                if mdp.board[i][j] == 'WALL':
                    continue
                if (i, j) in mdp.terminal_states:
                    utility_board[i][j] = float(mdp.board[i][j])
                    continue
                curr_state_policy = policy[i][j]  # get given policy
                reward = mdp.board[i][j]  # get R(S)=reward
                transition_func = mdp.transition_function[curr_state_policy]
                for index in range(4):
                    state = mdp.step((i, j), actions_num[index])
                    curr_sum += transition_func[index] * float(U[state[0]][state[1]])  # updating sum += (p(s')*U(s')
                expected_utility = float(reward) + gamma * curr_sum  # adding r(s)+gamma *sum (p(s')*U(s')
                utility_board[i][j] = expected_utility
                lambda_v = max(lambda_v, abs(utility_board[i][j] - U[i][j]))
        if lambda_v < (epsilon * (1 - discount_factor)) / discount_factor:
            return U


def policy_iteration(mdp, policy_init):
    # TODO:
    # Given the mdp, and the initial policy - policy_init
    # run the policy iteration algorithm
    # return: the optimal policy
    #

    # ====== YOUR CODE: ======
    unchanged = False
    while not unchanged:

        U = policy_evaluation(mdp, policy_init)

        new_policy = get_policy(mdp, U)
        if new_policy == policy_init:
            unchanged = True
        else:
            policy_init = new_policy

    return policy_init


"""For this functions, you can import what ever you want """


def get_all_policies(mdp, U):  # You can add more input parameters as needed
    # TODO:
    # Given the mdp, and the utility value U (which satisfies the Belman equation)
    # print / display all the policies that maintain this value
    # (a visualization must be performed to display all the policies)
    #
    # return: the number of different policies
    #

    # ====== YOUR CODE: ======
    raise NotImplementedError
    # ========================


def get_policy_for_different_rewards(mdp):  # You can add more input parameters as needed
    # TODO:
    # Given the mdp
    # print / displas the optimal policy as a function of r
    # (reward values for any non-finite state)
    #

    # ====== YOUR CODE: ======
    raise NotImplementedError
    # ========================
