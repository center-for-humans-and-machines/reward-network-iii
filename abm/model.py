from array_to_df import using_multiindex
import os
import numpy as np


def individual_learning(q_opt, d_myopic, d_optimal, k, mode, epsilon):
    q_opt_old = np.where(((mode == 'biased') | (mode == 'learned'))[:,np.newaxis], q_opt, np.ones_like(q_opt) * 0.5)
    q_opt_new = np.where((mode == 'learned')[:,np.newaxis], q_opt, np.ones_like(q_opt) * 0.5)
    
    # effective number of trails
    k_exploration = np.random.binomial(k, epsilon)
    k_opt = np.random.binomial(k_exploration, q_opt_old)
    k_myo = k_exploration - k_opt

    # probability of discovery
    d_opt_total = 1.0 - np.power(1.0 - d_optimal[:,np.newaxis], k_opt)
    d_myo_total = 1.0 - np.power(1.0 - d_myopic[:,np.newaxis], k_myo)

    u = np.random.rand(*q_opt.shape)
    discovered_optimal = (u < d_opt_total)
    discovered_myopic  = (~discovered_optimal) & (u < (d_opt_total + (1.0 - d_opt_total) * d_myo_total))
    random_solution = ~(discovered_myopic | discovered_optimal)

    # Update agent preferences when discovered optimal solution
    q_opt = np.where(discovered_myopic, 0, q_opt_new)
    q_opt = np.where(discovered_optimal, 1, q_opt_new)

    return q_opt

def demonstration(K_demonstration, epsilon, q_opt):
    n_random = np.random.binomial(K_demonstration, epsilon)
    n_optimal = np.random.binomial(K_demonstration - n_random, q_opt)
    n_myopic = K_demonstration - n_random - n_optimal
    return n_random, n_optimal, n_myopic

def get_rewards(n_random, n_optimal, n_myopic, sigma, R_random, R_optimal, R_myopic):
    base_reward = n_random * R_random + n_optimal * R_optimal + n_myopic * R_myopic
    noise = np.random.normal(0, sigma, size=base_reward.shape)
    return base_reward + noise

def get_teacher_indices(prev_rewards, n_teacher, N_gen):
    teacher_indices = np.zeros((M, N_gen), dtype=int)
    for m in range(prev_rewards.shape[0]): # For each replication
        for n in range(prev_rewards.shape[1]): # For each agent
            random_k_possible_teacher = np.random.choice(N_gen, n_teacher[m], replace=False) # Randomly select K possible teachers
            teacher_indices_within_k = np.argmax(prev_rewards[m, random_k_possible_teacher]) # Select the teacher with the highest average reward
            teacher_indices[m, n] = random_k_possible_teacher[teacher_indices_within_k] # Retrieve the teacher's index
    return teacher_indices


def social_learning(n_opt_t, n_myo_t, lambda_, M, N_gen):
    post_q_opt = (1 + n_opt_t) / (1 + n_opt_t + n_myo_t)
    is_learned = (np.random.rand(M, N_gen) < lambda_[:,np.newaxis])
    post_q_opt = np.where(is_learned, post_q_opt, 0.5)
    return post_q_opt




def grid_dict(params):
    keys = list(params)
    grids = np.meshgrid(*[params[k] for k in keys], indexing="ij")
    flat = [g.ravel() for g in grids]
    return {k: v for k, v in zip(keys, flat)}



def setup(conditions, d_optimal, d_myopic, lambda_, n_teacher, mode, n_rep, G, N_gen, N_mach, K_human, K_machine, q_opt_human, q_opt_machine, epsilon, eps_gamma):
    grid_d = {
        'condition': conditions,
        'd_optimal': d_optimal,
        'd_myopic': d_myopic,
        'lambda': lambda_,
        'n_teacher': n_teacher,
        'mode': mode,
        'replication': np.arange(n_rep),
        'epsilon': epsilon,
        'eps_gamma': eps_gamma
    }

    d = grid_dict(grid_d)
    M = len(d['replication'])
    is_machine = np.zeros((M, G, N_gen), dtype=bool)
    is_machine[conditions == 'human-machine',0,:N_mach] = True

    K_ind = np.ones((M, G, N_gen), dtype=int) * K_human
    K_ind[is_machine] = K_machine

    q_opt = np.ones((M, G, N_gen)) * q_opt_human
    q_opt[is_machine] = q_opt_machine

    e = np.ones((M, G, N_gen)) * epsilon[:,np.newaxis,np.newaxis]

    n_normal = np.empty((M, G, N_gen))
    n_optimal = np.empty((M, G, N_gen))
    n_myopic = np.empty((M, G, N_gen))
    rewards = np.empty((M, G, N_gen))
    d = {**d, 'n_normal': n_normal, 'n_optimal': n_optimal, 'n_myopic': n_myopic, 'rewards': rewards, 'q_opt': q_opt, 'e': e, 'K_ind': K_ind}
    return d



def run(d, output_path):
    cond = d['condition']
    d_opt = d['d_optimal']
    d_myo = d['d_myopic']
    n_t = d['n_teacher']
    l = d['lambda']
    m = d['mode']
    e_g = d['eps_gamma']
    epsilon = d['epsilon']
    n_normal = d['n_normal']
    n_optimal = d['n_optimal']
    n_myopic = d['n_myopic']
    rewards = d['rewards']
    q_opt = d['q_opt']
    e = d['e']
    K_ind = d['K_ind']


    for g in range(G):
        if g > 0:
            print("Select teacher")
            teacher_indices = get_teacher_indices(rewards[:,g-1], n_t)
            m_idx = np.arange(M)[:,np.newaxis]
            n_opt_t = n_normal[m_idx, g-1, teacher_indices]
            n_myo_t = n_myopic[m_idx, g-1, teacher_indices]
            print("Social learning")
            pre_q_opt = social_learning(n_opt_t, n_myo_t, l)
        else:
            pre_q_opt = np.ones_like(q_opt[:,g]) * 0.5
        print("Individual learning")
        post_q_opt = individual_learning(pre_q_opt, d_myo, d_opt, K_ind[:,g], m)
        q_opt[:,g] = post_q_opt
        print("Demonstration")
        n_normal[:,g], n_optimal[:,g], n_myopic[:,g] = demonstration(K_ind[:,g], epsilon, q_opt[:,g])
        rewards[:,g] = get_rewards(n_normal[:,g], n_optimal[:,g], n_myopic[:,g])

    df = using_multiindex(rewards, ['rep', 'gen', 'agent', 'problem'])
    df.to_parquet(os.path.join(output_path, 'rewards.parquet'))