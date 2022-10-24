import httpx
import pytest

from models.session import Session
from study_setup.generate_sessions import generate_sessions, create_trials


@pytest.mark.asyncio
@pytest.mark.slow  # 18 seconds
async def test_generate_sessions(default_client: httpx.AsyncClient,
                                 experiment_type='reward_network_iii',
                                 n_advise_per_session=5,
                                 n_generations=5,
                                 n_sessions_per_generation=20
                                 ):
    sessions = await Session.find().first_or_none()
    assert sessions is None

    await generate_sessions(experiment_type=experiment_type,
                            n_advise_per_session=n_advise_per_session,
                            n_generations=n_generations,
                            n_sessions_per_generation=n_sessions_per_generation)
    sessions = await Session.find().to_list(100)

    assert sessions is not None

    for s in sessions:
        assert s.experiment_type == "reward_network_iii"
        if s.generation != 0:
            # check the number of parents
            assert len(s.advise_ids) == n_advise_per_session

    # Clean up resources
    await Session.find().delete()


@pytest.mark.asyncio
async def test_create_trials(default_client: httpx.AsyncClient):
    n_consent = 1
    n_practice = 1
    n_soc_learning = 3
    n_ind = 3
    n_demonstration = 3
    n_w_strategy = 1
    n_debriefing = 1
    n_all_trials = n_consent + n_demonstration + n_w_strategy + n_debriefing
    n_all_trials += n_practice
    n_all_trials += n_soc_learning * n_demonstration * 3 + n_ind
    n_all_trials += 4  # 5 instructions: welcome, individual, demo, w_strategy

    session = create_trials(
        experiment_num=0,
        experiment_type='test',
        generation=0,
        session_idx=0,
        n_social_learning_trials=n_soc_learning,
        n_individual_trials=n_ind,
        n_demonstration_trials=n_demonstration)

    assert len(session.trials) == n_all_trials
    for t in session.trials:
        assert t.trial_type not in ['social_learning_selection',
                                    'social_learning']
        assert t.trial_type in ['consent', 'demonstration', 'written_strategy',
                                'debriefing', 'individual',
                                'instruction_welcome',
                                'instruction_individual',
                                'instruction_demonstration',
                                'instruction_written_strategy', 'practice']

    session = create_trials(
        experiment_num=0,
        experiment_type='test',
        generation=1,
        session_idx=0,
        n_social_learning_trials=n_soc_learning,
        n_individual_trials=n_ind,
        n_demonstration_trials=n_demonstration
    )

    # add n_all_trials because social_learning_selection counts as a trial
    # 2 instructions: social_learning_selection, social_learning
    assert len(session.trials) == n_all_trials + n_soc_learning + 2
    for t in session.trials:
        assert t.trial_type in [
            'consent', 'demonstration', 'written_strategy', 'debriefing',
            'individual', 'social_learning', 'social_learning_selection',
            'instruction_learning_selection', 'instruction_learning',
            'instruction_individual', 'instruction_demonstration',
            'instruction_written_strategy', 'instruction_welcome', 'practice']
