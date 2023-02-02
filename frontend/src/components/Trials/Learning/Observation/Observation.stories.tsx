import React, {useEffect} from 'react';

import {ComponentMeta, ComponentStory} from '@storybook/react';

import Observation from "./Observation";

import data from "../../../Network/examples";
import useNetworkContext, {NetworkContextProvider} from "../../../../contexts/NetworkContext";


export default {
    title: 'Trials/Observation',
    component: Observation,
    decorators: [
        (ComponentStory) => {
            return (
                <NetworkContextProvider>
                    <ComponentStory/>
                </NetworkContextProvider>
            );
        },
    ]
} as ComponentMeta<typeof Observation>;

const Template: ComponentStory<typeof Observation> = function (args) {
    const {networkState, networkDispatcher} = useNetworkContext();

    useEffect(() => {
        if (!networkState.network) {
            networkDispatcher({
                type: 'setNetwork',
                payload: {
                    network: {
                        edges: data[0].edges,
                        nodes: data[0].nodes
                    },
                    isTutorial: false
                }
            });
        }

    }, []);


    return (
        <>
            {networkState.network &&
                <Observation{...args}/>}
        </>


    );
};

export const Default = Template.bind({});

Default.args = {
    solution: [0, 1, 4, 3, 2, 1, 5, 2, 1],
};
