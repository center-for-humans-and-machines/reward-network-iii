import json
import random
from pathlib import Path

from typing import List

from models.network import Network
from models.session import Session
from models.trial import Trial, Solution, WrittenStrategy
from utils.utils import estimate_solution_score


async def generate_sessions(n_generations: int = 5,
                            n_sessions_per_generation: int = 20,
                            n_advise_per_session: int = 5,
                            experiment_type: str = 'reward_network_iii',
                            experiment_num: int = 0,
                            num_ai_players: int = 3,
                            seed: int = 4242):
    """
    Generate one experiment.
    """
    # Set random seed
    random.seed(seed)

    # check that n_sessions_per_generation is even
    # this is to ensure that there are an equal number of sessions with and
    # without AI player advisors
    assert n_sessions_per_generation % 2 == 0, \
        "n_sessions_per_generation must be even"

    # create sessions for the first generation
    # the last `num_ai_players` sessions are for AI players
    sessions_n_0 = await create_generation(
        0, n_sessions_per_generation, experiment_type, experiment_num,
        num_ai_players)

    sessions_n_0_with_ai = sessions_n_0[num_ai_players:]
    sessions_n_0_without_ai = sessions_n_0[
                              :n_sessions_per_generation - num_ai_players]

    # iterate over generations
    for generation in range(n_generations - 1):
        # create sessions for the next generation
        sessions_n_1 = await create_generation(
            generation + 1, n_sessions_per_generation, experiment_type,
            experiment_num)

        # split sessions into two streams (with and without AI player
        # advisors or offsprings of AI player advisors)
        sessions_n_1_with_ai = sessions_n_1[n_sessions_per_generation // 2:]
        await create_connections(sessions_n_0_with_ai,
                                 sessions_n_1_with_ai,
                                 n_advise_per_session)

        sessions_n_1_without_ai = sessions_n_1[:n_sessions_per_generation // 2]
        await create_connections(sessions_n_0_without_ai,
                                 sessions_n_1_without_ai,
                                 n_advise_per_session)

        # now sessions_n_0 is the previous generation
        # NOTE: the very first generation is different from the rest
        sessions_n_0_with_ai = sessions_n_1_with_ai
        sessions_n_0_without_ai = sessions_n_1_without_ai


async def create_connections(gen0, gen1, n_advise_per_session):
    # randomly link sessions of the previous generation to the sessions of
    # the next generation
    for s_n_1 in gen1:
        # get n numbers between 0 and len(gen0) - 1 without replacement
        advise_src = random.sample(range(len(gen0)), n_advise_per_session)
        advise_ids = []
        for i in advise_src:
            advise_ids.append(gen0[i].id)
            # record children of the session
            gen0[i].child_ids.append(s_n_1.id)
            await gen0[i].save()

        s_n_1.advise_ids = advise_ids

        # remove AI from the count of unfinished parents
        n_ai_advisors = sum([1 for i in advise_src if gen0[i].ai_player])
        s_n_1.unfinished_parents = len(advise_ids) - n_ai_advisors
        await s_n_1.save()


async def create_generation(generation: int,
                            n_sessions_per_generation: int,
                            experiment_type: str,
                            experiment_num: int,
                            num_ai_players: int = 0
                            ) -> List[Session]:
    sessions = []
    for session_idx in range(n_sessions_per_generation - num_ai_players):
        session = create_trials(experiment_num, experiment_type,
                                generation, session_idx)
        # save session
        await session.save()
        sessions.append(session)

    # if there are AI players, create sessions for them
    if num_ai_players > 0:
        for session_idx in range(n_sessions_per_generation - num_ai_players,
                                 n_sessions_per_generation):
            session = create_ai_trials(experiment_num, experiment_type,
                                       generation, session_idx)
            # save session
            await session.save()
            sessions.append(session)

    return sessions


def create_trials(experiment_num: int, experiment_type: str,
                  generation: int, session_idx: int,
                  n_social_learning_trials: int = 3,
                  n_individual_trials: int = 3,
                  n_demonstration: int = 3) -> Session:
    """
    Generate one session.
    """
    # load all networks
    network_data = json.load(open(Path('data') / 'train_viz.json'))
    trial_n = 0

    # Consent form
    trials = [Trial(id=trial_n, trial_type='consent')]
    trial_n += 1

    # Social learning trials (not relevant for the very first generation)
    if generation > 0:
        for i in range(n_social_learning_trials):
            # Social learning selection
            trials.append(Trial(id=trial_n,
                                trial_type='social_learning_selection'))
            trial_n += 1

            # Social learning
            trials.append(Trial(id=trial_n, trial_type='social_learning',
                                social_learning_type='observation'))
            trial_n += 1
            trials.append(Trial(id=trial_n, trial_type='social_learning',
                                social_learning_type='repeat'))
            trial_n += 1
            trials.append(Trial(id=trial_n, trial_type='social_learning',
                                social_learning_type='tryyourself'))
            trial_n += 1
    else:
        # Replace social learning trials with individual trials for the very
        # first generation
        n_individual_trials += n_social_learning_trials * 3

    # Individual trials

    for i in range(n_individual_trials):
        # individual trial
        trial = Trial(
            trial_type='individual',
            id=trial_n,
            network=Network.parse_obj(
                network_data[random.randint(0, network_data.__len__() - 1)]),
        )
        # update the starting node
        trial.network.nodes[trial.network.starting_node].starting_node = True
        trials.append(trial)
        trial_n += 1

    for i in range(n_demonstration):
        # demonstration trial
        dem_trial = Trial(
            id=trial_n,
            trial_type='demonstration',
            network=Network.parse_obj(
                network_data[random.randint(0, network_data.__len__() - 1)]),
        )
        # update the starting node
        dem_trial.network.nodes[
            dem_trial.network.starting_node].starting_node = True
        trials.append(dem_trial)
        trial_n += 1

    # Written strategy
    trials.append(Trial(id=trial_n, trial_type='written_strategy'))
    trial_n += 1

    # Debriefing
    trials.append(Trial(id=trial_n, trial_type='debriefing'))
    trial_n += 1

    # create session
    session = Session(
        experiment_num=experiment_num,
        experiment_type=experiment_type,
        generation=generation,
        session_num_in_generation=session_idx,
        trials=trials,
        available=True if generation == 0 else False
    )
    # Add trials to session
    session.trials = trials
    return session


def create_ai_trials(experiment_num, experiment_type, generation,
                     session_idx, n_demonstration=3):
    network_data = json.load(open(Path('data') / 'train_viz.json'))
    solutions = json.load(
        open(Path('data') / 'solution_moves_take_first_loss_viz.json'))
    trials = []

    # Demonstration trial
    for i in range(n_demonstration):
        network = Network.parse_obj(
            network_data[random.randint(0, network_data.__len__() - 1)])
        moves = [s for s in solutions if s['network_id'] == network.network_id][
            0]['moves']

        dem_trial = Trial(
            id=i,
            trial_type='demonstration',
            network=network,
            solution=Solution(
                moves=moves,
                score=estimate_solution_score(network, moves)
            )
        )
        # update the starting node
        dem_trial.network.nodes[
            dem_trial.network.starting_node].starting_node = True
        trials.append(dem_trial)

    # Written strategy
    trials.append(Trial(
        id=n_demonstration + 1,
        trial_type='written_strategy',
        written_strategy=WrittenStrategy(strategy=''))
    )

    session = Session(
        experiment_num=experiment_num,
        experiment_type=experiment_type,
        generation=generation,
        session_num_in_generation=session_idx,
        trials=trials,
        available=False,
        ai_player=True,
        finished=True
    )
    return session
