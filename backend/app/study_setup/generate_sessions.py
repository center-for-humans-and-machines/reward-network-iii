import json
import random
from pathlib import Path
from typing import List

# from beanie import PydanticObjectId
from beanie.odm.operators.update.general import Set

from models.config import ExperimentSettings
from models.network import Network
from models.session import Session
from models.subject import Subject
from models.trial import Trial, Solution, WrittenStrategy
from utils.utils import estimate_solution_score, estimate_average_player_score

network_data = None
solutions = None
MAX_STEPS = 10

# load all ai solutions


def get_net_solution(solution_type="loss"):
    # get networks list from the global variable
    global network_data

    # load the network again if all the previous networks have been used
    # TODO: maybe change this behavior later
    if len(network_data) == 0:
        raise Exception("All networks have been used")

    # pop a network from the list of networks
    network_raw = network_data.pop(0)
    network_data.append(network_raw)

    # parse the network
    network = Network.parse_obj(network_raw)

    # get the solution for the network
    solution = solutions[solution_type][network.network_id]

    # for some reason the first move is always 0, so we need to replace it
    solution["moves"][0] = network.starting_node

    return network, solution["moves"]


def reset_networks(config: ExperimentSettings):
    global network_data, solutions
    # load all networks
    network_data = json.load(open(Path(config.networks_path) / "solution__valid_networks.json"))
    solutions = json.load(open(Path(config.networks_path) / "solution__take_loss.json"))
    solutions_myopic = json.load(open(Path(config.networks_path) / "solution__myopic.json"))
    solutions_m1 = json.load(open(Path(config.networks_path) / "machine_solutions" / "0.json"))
    solutions_m2 = json.load(open(Path(config.networks_path) / "machine_solutions" / "1.json"))
    solutions_m3 = json.load(open(Path(config.networks_path) / "machine_solutions" / "2.json"))
    solutions_dict = {s['network_id']: s for s in solutions}
    solutions_myopic_dict = {s['network_id']: s for s in solutions_myopic}
    solutions_m1_dict = {s['network_id']: s for s in solutions_m1}
    solutions_m2_dict = {s['network_id']: s for s in solutions_m2}
    solutions_m3_dict = {s['network_id']: s for s in solutions_m3}
    solutions = {
        "loss": solutions_dict,
        "myopic": solutions_myopic_dict,
        "machine_0": solutions_m1_dict,
        "machine_1": solutions_m2_dict,
        "machine_2": solutions_m3_dict,
    }
    # randomize the order of the networks
    random.shuffle(network_data)
    print("Networks loaded", len(network_data), flush=True)


async def generate_experiment_sessions(config: ExperimentSettings):
    if config.rewrite_previous_data:
        await Session.find(Session.experiment_type == config.experiment_type).delete()
        await Subject.find(Session.experiment_type == config.experiment_type).delete()

    # find all sessions for this experiment
    sessions = await Session.find(
        Session.experiment_type == config.experiment_type
    ).first_or_none()

    if sessions is None:
        # if the database is empty, generate sessions
        for replication in range(config.n_session_tree_replications):
            reset_networks(config)
            await generate_sessions(experiment_num=replication, config=config)

    # set all sessions of the first generation to available
    await Session.find(
        Session.experiment_type == config.experiment_type,
        Session.generation == 0,
        Session.finished == False,
        Session.replaced == False,
        Session.expired == False,
        Session.ai_player == False,
    ).update(Set({Session.available: True}))


async def generate_sessions(
    experiment_num: int,
    config: ExperimentSettings,
):
    """
    Generate one experiment.
    """
    # create sessions for the first generation
    # the last `num_ai_players` sessions are for AI players

    previous_sessions = None

    for generation in range(config.n_generations):
        print(f"Generating generation {generation}", flush=True)
        sessions = await create_generation(
            generation=generation,
            experiment_num=experiment_num,
            config=config,
        )
        print(f"Generation {generation} created", flush=True)
        if previous_sessions is not None:
            for condition in config.conditions:
                possible_parents = [
                    s
                    for s in previous_sessions
                    if (s.condition == condition) or (s.condition is None)
                ]
                possible_children = [
                    s
                    for s in sessions
                    if (s.condition == condition) or (s.condition is None)
                ]
                print(f"Creating connections from {len(possible_parents)} parents to {len(possible_children)}", flush=True)
                await create_connections(
                    possible_parents,
                    possible_children,
                    config.n_advise_per_session,
                )

        previous_sessions = sessions


async def create_connections(gen0, gen1, n_advise_per_session):
    # randomly link sessions of the previous generation to the sessions of
    # the next generation
    for s_n_1 in gen1:
        # get n numbers between 0 and len(gen0) - 1 without replacement
        advise_src = random.sample(range(len(gen0)), n_advise_per_session)
        advise_ids = []
        for i in advise_src:
            # advise_ids.append(gen0[i].id)
            # record children of the session
            gen0[i].child_ids.append(s_n_1.id)
            await gen0[i].save()
            if gen0[i].ai_player:
                advise_ids.append(gen0[i].id)

        s_n_1.advise_ids = advise_ids
        s_n_1.available = s_n_1.available or len(advise_ids) >= n_advise_per_session
        await s_n_1.save()


async def create_generation(
    generation: int,
    experiment_num: int,
    config: ExperimentSettings,
) -> List[Session]:
    machines = []
    # compute conditions
    if generation == 0:
        if len(config.conditions) == 1:
            conditions = [config.conditions[0]] * config.n_sessions_per_generation
            if config.conditions[0] == "w_ai":
                machines = list(range(config.n_ai_players))
                machines += [None] * (config.n_sessions_per_generation - config.n_ai_players)
            else:
                machines = [None] * config.n_sessions_per_generation
        elif len(config.conditions) == 2 and config.shared_first_generation:
            conditions = [config.conditions[0]] * config.n_ai_players
            machines = list(range(config.n_ai_players)) if config.conditions[0] == "w_ai" else [None] * config.n_ai_players
            conditions += [None] * (
                (config.n_sessions_per_generation // len(config.conditions))
                - config.n_ai_players
            )
            machines += [None] * (
                (config.n_sessions_per_generation // len(config.conditions))
                - config.n_ai_players
            )
            conditions += [config.conditions[1]] * config.n_ai_players
            machines += list(range(config.n_ai_players)) if config.conditions[1] == "w_ai" else [None] * config.n_ai_players
        elif len(config.conditions) == 2 and not config.shared_first_generation:
            conditions = [
                c for c in config.conditions for _ in range(config.n_sessions_per_generation // len(config.conditions))
            ]
            machines = list(range(config.n_ai_players)) if config.conditions[0] == "w_ai" else [None] * config.n_ai_players
            machines += [None] * (config.n_sessions_per_generation - config.n_ai_players)
            machines += list(range(config.n_ai_players)) if config.conditions[1] == "w_ai" else [None] * config.n_ai_players
            machines += [None] * (config.n_sessions_per_generation - config.n_ai_players)
        else:
            raise Exception("Only 1 or 2 conditions are supported")
    else:
        assert (
            config.n_sessions_per_generation % 2 == 0
        ), "n_sessions_per_generation must be even"
        n_sessions_per_condition = config.n_sessions_per_generation // len(
            config.conditions
        )
        conditions = [
            c for c in config.conditions for _ in range(n_sessions_per_condition)
        ]
        machines = [None] * config.n_sessions_per_generation

    sessions = []
    for session_idx, (condition, machine_idx) in enumerate(zip(conditions, machines)):
        session = create_trials(
            experiment_num=experiment_num,
            generation=generation,
            condition=condition,
            config=config,
            session_idx=session_idx,
            machine_idx=machine_idx,
        )
        # save session
        await session.save()
        sessions.append(session)
    return sessions


def add_consent_trail(trials):
    return [*trials, Trial(
            id=len(trials),
            trial_type="consent",
            redirect_url="https://www.prolific.co/",
        )]

def add_practice_trail(trials):
    return [*trials, Trial(id=len(trials), trial_type="practice", trial_title="Tutorial")]

def add_instruction_trail(trials, instruction_type, title=None):
    return [*trials, Trial(id=len(trials), trial_type="instruction",
                  instruction_type=instruction_type, trial_title=title if title else "Instructions")]


def add_self_practice_trail(trials, i, config):
    net, _ = get_net_solution()
    trial = Trial(
        trial_type="individual",
        id=len(trials),
        network=net,
        is_practice=True,
        trial_title=f"Self Practice | Network {i+1} of {config.n_practice_trials}",
    )
    trial.network.nodes[trial.network.starting_node].starting_node = True
    return [*trials, trial]

def add_written_strategy_trail(trials, written_strategy=None):
    return [*trials, Trial(id=len(trials), trial_type="written_strategy", written_strategy=written_strategy)]

def add_social_learning_selection_trail(trials, block_idx, config):
    return [*trials, Trial(id=len(trials), trial_type="social_learning_selection", social_learning_block_idx=block_idx, trial_title=f"Learning Phase | Select Teacher")]

def add_social_learning_network_gen0(trials, block_idx, network_idx, is_human, simulated_subject, machine_idx, config: ExperimentSettings):
    if is_human:
        net, _ = get_net_solution()
        solution = None
    else:
        if simulated_subject:
            solution_type = "myopic"
        else:
            solution_type = f"machine_{machine_idx}"
        net, moves = get_net_solution(solution_type)
        solution = Solution(
            moves=moves,
            score=estimate_solution_score(net, moves, n_steps=MAX_STEPS),
            solution_type=solution_type,
        )
        assert solution.score > -100_000, "invalid move sequence"

    n_trails = len([t for t in config.social_learning_trials if t in ['try_yourself']])

    for iii in range(n_trails):
        title_postfix = "" if n_trails == 1 else f" | Trial {iii+1} of {n_trails}"
        total_networks = config.n_social_learning_blocks * config.n_social_learning_networks_per_block
        total_network_idx = block_idx * config.n_social_learning_networks_per_block + network_idx
        trial = Trial(
            trial_type="individual",
            id=len(trials),
            network=net,
            is_practice=False,
            solution=solution,
            social_learning_block_idx=block_idx,
            block_network_idx=network_idx,
            trial_title=f"Learning Phase | Network {total_network_idx + 1} of {total_networks}{title_postfix}"
        )
        # update the starting node
        trial.network.nodes[
            trial.network.starting_node
        ].starting_node = True
        trials = [*trials, trial]
    return trials


def add_social_learning_network(trials, block_idx, network_idx, config: ExperimentSettings):
    trial_type_titles = {
        "repeat": "Repeat the Teacher's Solution",
        "try_yourself": "Try Yourself",
        "observation": "Observe the Teacher's Solution",
    }
    for i, trial_type in enumerate(config.social_learning_trials):
        total_networks = config.n_social_learning_blocks * config.n_social_learning_networks_per_block
        total_network_idx = block_idx * config.n_social_learning_networks_per_block + network_idx
        trials.append(
            Trial(
                id=len(trials),
                trial_type=trial_type,
                is_practice=trial_type in ["observation"],
                social_learning_block_idx=block_idx,
                block_network_idx=network_idx,
                last_trial_for_current_example=(i == len(config.social_learning_trials) - 1),
                trial_title=f"Learning Phase | Network {total_network_idx + 1} of {total_networks} | {trial_type_titles[trial_type]}",
            )
        )
    return trials

def add_demonstration_trail(trials, is_human, simulated_subject, network_idx, machine_idx, config: ExperimentSettings):
    if is_human:
        net, _ = get_net_solution()
        solution = None
    else:
        if simulated_subject:
            solution_type = "myopic"
        else:
            solution_type = f"machine_{machine_idx}"
        net, moves = get_net_solution(solution_type)
        solution = Solution(
            moves=moves,
            score=estimate_solution_score(net, moves, n_steps=MAX_STEPS),
            solution_type=solution_type,
        )
        assert solution.score > -100_000, "invalid move sequence"
    # demonstration trial
    dem_trial = Trial(
        id=len(trials),
        trial_type="demonstration",
        network=net,
        solution=solution,
        block_network_idx=network_idx,
        trial_title=f"Teaching Phase | Network {network_idx + 1} of {config.n_demonstration_trials}",
    )
    # update the starting node
    dem_trial.network.nodes[dem_trial.network.starting_node].starting_node = True
    trials.append(dem_trial)
    return trials

def add_exit_trails(trials, config):
    if not config.main_only:
        trials.append(Trial(id=len(trials), trial_type="post_survey"))

    # Debriefing
    trials.append(
        Trial(
            id=len(trials),
            trial_type="debriefing",
            redirect_url=config.redirect_url,
        )
    )
    return trials


def create_trials(
    experiment_num: int,
    session_idx: int,
    condition: str,
    generation: int,
    machine_idx: int = None,
    config: ExperimentSettings = None,
) -> Session:
    """
    Generate one session.
    :param redirect_url: URL to redirect to after the experiment is finished
    """
    assert config.n_demonstration_trials > 0, "n_demonstration_trials must be > 0"

    global network_data
    if network_data is None:
        reset_networks(config)
    is_ai = ((generation == 0) and config.simulate_humans) or machine_idx is not None
    is_human = not is_ai
    simulated_subject = is_ai and machine_idx is None

    trials = []

    if is_human and not config.main_only:
        trials = add_consent_trail(trials)
        trials = add_instruction_trail(trials, "welcome", "Welcome to the Experiment!")
        trials = add_practice_trail(trials)

        # Individual trials for practice
        for i in range(config.n_practice_trials):
            if i == 0:
                trials = add_instruction_trail(trials, "practice_rounds", "Self Practice")
            trials = add_self_practice_trail(trials, i, config)

        trials = add_written_strategy_trail(trials)

    # Social learning blocks
    for block_idx in range(config.n_social_learning_blocks):
        # social learning selection
        if is_human and (generation > 0):
            if block_idx == 0:
                trials = add_instruction_trail(trials, "learning_selection", "Learning Phase | Select Teacher")
            trials = add_social_learning_selection_trail(trials, block_idx, config)

        # instruction before learning
        if is_human and block_idx == 0:
            instruction_type = "pre_social_learning_gen0" if generation == 0 else "pre_social_learning"
            trials = add_instruction_trail(trials, instruction_type, "Learning Phase | Instructions")

        # run social learning blocks
        for network_idx in range(config.n_social_learning_networks_per_block):
            if generation == 0:
                trials = add_social_learning_network_gen0(trials, block_idx, network_idx, is_human, simulated_subject, machine_idx, config)
            else:
                trials = add_social_learning_network(trials, block_idx, network_idx, config)

    trials = add_written_strategy_trail(trials, None if is_human else WrittenStrategy(strategy=""))
    if is_human:
        trials = add_instruction_trail(trials, "demonstration", "Teaching Phase")

    for network_idx in range(config.n_demonstration_trials):
        trials = add_demonstration_trail(trials, is_human, simulated_subject, network_idx, machine_idx, config)

    if is_human:
        trials = add_exit_trails(trials, config)

    # create session
    session = Session(
        config_id=config.id,
        experiment_num=experiment_num,
        experiment_type=config.experiment_type,
        priority=-(experiment_num * config.n_generations + generation + random.random()),
        # priority=-(experiment_num * 0 + generation + random.random()),
        generation=generation,
        session_num_in_generation=session_idx,
        trials=trials,
        available=(generation == 0) and is_human,
        ai_player=is_ai,
        finished=is_ai,
        completed=is_ai,
        condition=condition,
        simulated_subject=simulated_subject,
    )
    if is_ai:
        session.average_score = estimate_average_player_score(session)
    return session
