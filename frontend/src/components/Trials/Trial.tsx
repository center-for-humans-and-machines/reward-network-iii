import React, {useEffect, useState} from "react";
import {useTrialAPI} from "../../apis";
import Header from "../Header";
import WrittenStrategy from "./WrittenStrategy";
import ConsentForm from "./Intro/Consent";
import Selection from "./SocialLearning/Selection";
import IndividualTrial from "./IndividualTrial";
import ObservationTrial from "./SocialLearning/Observation";
import RepeatTrial from "./SocialLearning/Repeat";
import TryYourselfTrial from "./SocialLearning/TryYourself";
import {Trial} from "../../apis/apiTypes";

interface TrialInterface {
    nextTrialHandler: () => null;
}

const Trial: React.FC<TrialInterface> = (props) => {
    const {trial, loading, error, axiosGetRequest, axiosPostRequest} = useTrialAPI();
    const [trialType, setTrialType] = useState<string>('');
    const [trialNumber, setTrialNumber] = useState<number>(0);
    const [socialLearningStage, setSocialLearningStage] = useState<number>(1);
    const [moves, setMoves] = useState<number[]>([]);

    useEffect(() => {axiosGetRequest({method: 'GET'})}, [])

    useEffect(() => {
        setTrialType(trial?.trial_type);
        setTrialNumber(trial?.id);
    }, [trial])

    const OnNextTrial = () => {
        axiosPostRequest({
            method: 'POST',
            data: {
                moves: moves
            }
        }).then(
            () => {
                props.nextTrialHandler();
                axiosGetRequest({method: 'GET'});
            }
        )
    }

    const renderTrial = (type: string, data: Trial) => {
        switch (type) {
            case 'consent':
                return <ConsentForm
                    onClickAgreeHandler={OnNextTrial}
                    onClickDisagreeHandler={() => null}
                />;
            case 'social_learning_selection':
                return <Selection
                    advisors={
                        data.advisor_selection.scores.map((score: number, inx: number) => {
                            return {
                                advisorInx: inx,
                                averageScore: score
                            }
                        })
                    }
                    onClickHandler={OnNextTrial}
                />;
            case 'social_learning':
                if (socialLearningStage === 1) {
                    setSocialLearningStage(2);
                    return <ObservationTrial
                        nodes={data.network.nodes}
                        edges={data.network.edges}
                        moves={data.advisor.solution.moves}
                        teacherId={1}  // TODO: set correct teacher id
                        onNextTrialHandler={OnNextTrial}
                    />;
                } else if (socialLearningStage === 2) {
                    setSocialLearningStage(3);
                    return <RepeatTrial
                        nodes={data.network.nodes}
                        edges={data.network.edges}
                        moves={data.advisor.solution.moves}
                        teacherId={1}  // TODO: set correct teacher id
                        onNextTrialHandler={OnNextTrial}
                    />;
                } else {
                    setSocialLearningStage(1);
                    return <TryYourselfTrial
                        nodes={data.network.nodes}
                        edges={data.network.edges}
                        moves={data.advisor.solution.moves}
                        teacherId={1}  // TODO: set correct teacher id
                        onNextTrialHandler={OnNextTrial}
                    />;
                }
            case  'individual':
                return <IndividualTrial
                    nodes={data.network.nodes}
                    edges={data.network.edges}
                    onNextTrialHandler={OnNextTrial}
                />;
            case 'demonstration':
                return <IndividualTrial
                    timer={2 * 60}
                    nodes={data.network.nodes}
                    edges={data.network.edges}
                    onNextTrialHandler={OnNextTrial}
                />;
            case  'written_strategy':
                return <WrittenStrategy onClickContinue={OnNextTrial}/>;
            default:
                return <> </>;
        }
    }

    return (
        <>
            {error && (console.log(error))}
            {!loading && !error ?
                (
                    <>
                        <Header title={"Trial " + trialNumber + ": " + trialType}/>
                        {renderTrial(trialType, trial)}
                    </>
                ) : (
                    // TODO: add loading screen
                <div>Loading...</div>
                )
            }
        </>
    );


};


export default Trial;