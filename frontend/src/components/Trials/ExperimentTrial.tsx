import React, {FC, useEffect, useState} from "react";
import {useMutation, useQuery} from "react-query";

// Contexts & APIs
import useNetworkContext from "../../contexts/NetworkContext";
import {NETWORK_ACTIONS} from "../../reducers/NetworkReducer";
import useSessionContext from "../../contexts/SessionContext";
import {SESSION_ACTIONS} from "../../reducers/SessionReducer";
import {getTrial, postTrial, postTrialType} from "../../apis/TrialAPI";
import {useSearchParamsContext} from "../App";
import {
    AdvisorSelection,
    SessionError,
    Trial,
    TrialError,
    TrialSaved,
} from "../../apis/apiTypes";

// Data
import {
    edges as practiceEdges,
    nodes as practiceNodes,
} from "./Practice/PracticeData";

import {exampleData} from "./StaticExample";

const exampleData1 = exampleData[0];

// Trials
import {
    ConsentTrial,
    DebriefingTrial,
    DemonstrationTrial,
    IndividualTrial,
    InstructionTrial,
    ObservationTrial,
    PostSurveyTrial,
    PracticeTrial,
    RepeatTrial,
    SelectionTrial,
    TryYourselfTrial,
    WrittenStrategyTrial,
} from "./Trials";
import WaitForNextTrialScreen from "./WaitForNextTrialScreen";
import ErrorMessage from "./ErrorMessage";

export const TRIAL_TYPE = {
    // before the experiment
    CONSENT: "consent",
    INSTRUCTION: "instruction",
    PRACTICE: "practice",
    // Social learning selection
    SOCIAL_LEARNING_SELECTION: "social_learning_selection",
    // Network trials
    OBSERVATION: "observation",
    REPEAT: "repeat",
    TRY_YOURSELF: "try_yourself",
    INDIVIDUAL: "individual",
    DEMONSTRATION: "demonstration",
    // after the experiment
    WRITTEN_STRATEGY: "written_strategy",
    POST_SURVEY: "post_survey",
    DEBRIEFING: "debriefing",
};

const ExperimentTrial: FC = () => {
    const {networkState, networkDispatcher} = useNetworkContext();
    const {sessionState, sessionDispatcher} = useSessionContext();
    const [trialId, setTrialId] = useState(0);
    const [isDataReady, setIsDataReady] = useState(false);

    const onTrialStart = (data: Trial | SessionError) => {
        data = data as Trial;

        // update session state
        sessionDispatcher({
            type: SESSION_ACTIONS.SET_CURRENT_TRIAL,
            payload: {
                currentTrialId: data.id,
                currentTrialType: data.trial_type,
                is_practice: data.is_practice,
                last_trial_for_current_example: data.last_trial_for_current_example,
            },
        });

        switch (data.trial_type) {
            case TRIAL_TYPE.PRACTICE:
                networkDispatcher({
                    type: NETWORK_ACTIONS.SET_NETWORK,
                    payload: {
                        network: {edges: practiceEdges, nodes: practiceNodes},
                        isPractice: true,
                    },
                });
                break;
            case TRIAL_TYPE.SOCIAL_LEARNING_SELECTION:
                sessionDispatcher({
                    type: SESSION_ACTIONS.SET_ADVISORS,
                    payload: {advisors: data.advisor_selection as AdvisorSelection},
                });
                break;
            case TRIAL_TYPE.INDIVIDUAL:
            case TRIAL_TYPE.DEMONSTRATION:
                networkDispatcher({
                    type: NETWORK_ACTIONS.SET_NETWORK,
                    payload: {
                        network: {edges: data.network.edges, nodes: data.network.nodes},
                        isPractice: data.is_practice,
                        trialType: data.trial_type,
                    },
                });
                break;
            case TRIAL_TYPE.TRY_YOURSELF:
            case TRIAL_TYPE.OBSERVATION:
            case TRIAL_TYPE.REPEAT:
                const isRepeat = data.trial_type === TRIAL_TYPE.REPEAT;
                networkDispatcher({
                    type: NETWORK_ACTIONS.SET_NETWORK,
                    payload: {
                        network: {edges: data.network.edges, nodes: data.network.nodes},
                        solution: data.advisor.solution,
                        isPractice: data.is_practice,
                        // teacherComment: data.advisor && data.advisor.written_strategy,
                        // show comment tutorial only for the first observation trial
                        // commentTutorial: data.trial_type === TRIAL_TYPE.OBSERVATION &&
                        //     sessionState.showTutorialInCurrentTrial,
                        wrongRepeatPunishment: isRepeat ? -100 : 0,
                        correctRepeatReward: isRepeat ? 100 : 0,
                        trialType: data.trial_type,
                    },
                });
                break;
            case TRIAL_TYPE.INSTRUCTION:
                // if this is instruction before the first individual trial clean up the total points
                if (data.instruction_type === "individual")
                    sessionDispatcher({type: SESSION_ACTIONS.CLEAN_TOTAL_POINTS});
                break;
            default:
                break;
        }
    };

    const onTrialEnd = () => {
        if (
            [
                TRIAL_TYPE.INDIVIDUAL,
                TRIAL_TYPE.TRY_YOURSELF,
                TRIAL_TYPE.REPEAT,
            ].includes(sessionState.currentTrialType)
        ) {
            sessionDispatcher({
                type: sessionState.isPractice
                    ? SESSION_ACTIONS.UPDATE_PRACTICE_POINTS
                    : SESSION_ACTIONS.UPDATE_TOTAL_POINTS,
                payload: {
                    points: networkState.points ? networkState.points : 0,
                    // NOTE: the max number of steps is assumed to be 10
                    missingSteps:
                        sessionState.currentTrialType === TRIAL_TYPE.REPEAT
                            ? 0
                            : 10 - networkState.step,
                },
            });
        }
    };

    // run the function once
    useEffect(() => {
        // clear all the info in the local storage
        onTrialStart(exampleData1.trials[trialId] as unknown as Trial);
        setIsDataReady(true);
    }, []);

    const submitResults = (result: postTrialType["trialResults"]) => {
        onTrialEnd();
        onTrialStart(exampleData1.trials[trialId + 1] as unknown as Trial);
        setTrialId(trialId + 1);
        setIsDataReady(true);
    }

    const trialData = exampleData1.trials[trialId] as unknown as Trial;

    if (isDataReady) {
        switch (trialData.trial_type) {
            case TRIAL_TYPE.CONSENT:
                return <ConsentTrial endTrial={submitResults} data={trialData}/>;
            case TRIAL_TYPE.INSTRUCTION:
                return <InstructionTrial endTrial={submitResults} data={trialData}/>;
            case TRIAL_TYPE.PRACTICE:
                return <PracticeTrial endTrial={submitResults} data={trialData}/>;
            case TRIAL_TYPE.SOCIAL_LEARNING_SELECTION:
                return <SelectionTrial endTrial={submitResults} data={trialData}/>;
            case TRIAL_TYPE.OBSERVATION:
                return <ObservationTrial endTrial={submitResults} data={trialData}/>;
            case TRIAL_TYPE.REPEAT:
                return <RepeatTrial endTrial={submitResults} data={trialData}/>;
            case TRIAL_TYPE.TRY_YOURSELF:
                return <TryYourselfTrial endTrial={submitResults} data={trialData}/>;
            case TRIAL_TYPE.INDIVIDUAL:
                return <IndividualTrial endTrial={submitResults} data={trialData}/>;
            case TRIAL_TYPE.DEMONSTRATION:
                return <DemonstrationTrial endTrial={submitResults} data={trialData}/>;
            case TRIAL_TYPE.WRITTEN_STRATEGY:
                return <WrittenStrategyTrial endTrial={submitResults} data={trialData}/>;
            case TRIAL_TYPE.POST_SURVEY:
                return <PostSurveyTrial endTrial={submitResults} data={trialData}/>;
            case TRIAL_TYPE.DEBRIEFING:
                return <DebriefingTrial endTrial={submitResults} data={trialData}/>;
            default:
                return <> </>;
        }
    } else {
        return <WaitForNextTrialScreen/>;
    }
};

export default ExperimentTrial;
