from pathlib import Path
from typing import List

from pyvis.network import Network

from models.session import Session

ROOT = Path(__file__).parent


async def create_sessions_network(experiment_type: str = 'reward_network_iii',
                                  experiment_num: int = 0) -> Path:
    sessions = await Session.find(
        Session.experiment_num == experiment_num,
        Session.experiment_type == experiment_type
    ).sort(+Session.generation).to_list()

    net = Network(height='800px', width='100%', directed=True, layout=True)
    for session in sessions:
        g = session.generation
        s_num = session.session_num_in_generation
        label = f'{g}_{s_num}'

        if session.available:
            color = '#85D4E3'
        else:
            if session.finished:
                color = '#81A88D'
            else:
                color = '#FAD77B'

        net.add_node(str(session.id), label, shape='circle', level=g, x=s_num,
                     color=color)
        adv = session.advise_ids
        if adv is not None:
            for a in adv:
                advise_session = await Session.find_one(Session.id == a)
                if session.available & advise_session.available:
                    opacity = 1
                else:
                    opacity = 0.5
                net.add_edge(str(advise_session.id), str(session.id),
                             opacity=opacity)
    net.set_options(open(ROOT / 'graph_settings.json').read())
    path = ROOT / 'tmp'
    path.mkdir(exist_ok=True)
    file = path / f'study_{experiment_type}_{experiment_num}_overview.html'
    net.save_graph(str(file))
    return file
