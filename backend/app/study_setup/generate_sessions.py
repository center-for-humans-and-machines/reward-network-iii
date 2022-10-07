import json
import random
from pathlib import Path

from typing import List

from models.network import Network
from models.session import Session
from models.trial import Trial


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
        await s_n_1.save()


async def create_generation(generation: int,
                            n_sessions_per_generation: int,
                            experiment_type: str,
                            experiment_num: int,
                            num_ai_players: int = 0
                            ) -> List[Session]:
    sessions = []
    for session_idx in range(n_sessions_per_generation - num_ai_players):
        session = await create_trials(experiment_num, experiment_type,
                                      generation, session_idx)
        # save session
        await session.save()
        sessions.append(session)

    # if there are AI players, create sessions for them
    if num_ai_players > 0:
        for session_idx in range(n_sessions_per_generation - num_ai_players,
                                 n_sessions_per_generation):
            session = await create_ai_trials(experiment_num, experiment_type,
                                             generation,
                                             session_idx)
            # save session
            await session.save()
            sessions.append(session)

    return sessions


async def create_trials(experiment_num, experiment_type, generation,
                        session_idx):
    """
    Generate one session.
    """
    network_data = json.load(open(Path('data') / 'train_viz.json'))
    trial_n = 0

    # Consent form
    trials = [Trial(id=trial_n, trial_type='consent')]
    trial_n += 1

    # Social learning
    # trials.append(Trial(
    #     trial_num_in_session=trial_n,
    #     trial_type='social_learning_selection'))
    # trial_n += 1

    # Individual trials
    n_individual_trials = 3
    for _ in range(n_individual_trials):
        # TODO: read Networks
        # create trial
        trial = Trial(
            trial_type='individual',
            id=trial_n,
            network=Network.parse_obj(
                network_data[random.randint(0, network_data.__len__() - 1)]),
        )
        # update the starting node
        trial.network.nodes[
            trial.network.starting_node].starting_node = True
        trials.append(trial)
        trial_n += 1

    # Demonstration trial
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
    trials.append(Trial(
        id=trial_n,
        trial_type='written_strategy'))
    trial_n += 1

    # Debriefing

    # create session
    # TODO: check if session already exists
    session = Session(
        experiment_num=experiment_num,
        experiment_type=experiment_type,
        generation=generation,
        session_num_in_generation=session_idx,
        trials=trials,
        available=True if generation == 0 else False,
    )
    # Add trials to session
    session.trials = trials
    return session


async def create_ai_trials(experiment_num, experiment_type, generation,
                           session_idx):
    # TODO: create AI player trials with solutions
    network_data = json.load(open(Path('data') / 'train_viz.json'))
    trial_n = 0

    # Demonstration trial
    dem_trial = Trial(
        id=trial_n,
        trial_type='demonstration',
        network=Network.parse_obj(
            network_data[random.randint(0, network_data.__len__() - 1)]),
    )

    session = Session(
        experiment_num=experiment_num,
        experiment_type=experiment_type,
        generation=generation,
        session_num_in_generation=session_idx,
        trials=[dem_trial],
        available=False,
        ai_player=True
    )
    return session
