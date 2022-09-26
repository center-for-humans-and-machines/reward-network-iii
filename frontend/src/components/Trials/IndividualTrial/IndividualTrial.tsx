import React, {useEffect, useState} from "react";
import {Box, Grid, Paper, Typography} from "@mui/material";
import DynamicNetwork from "../../Network/DynamicNetwork";
import {DynamicNetworkInterface} from "../../Network/DynamicNetwork/DynamicNetwork";
import Timer from "./Timer";


export interface IndividualTrialInterface extends DynamicNetworkInterface {
    /** Handle the end of the trial */
    onNextTrialHandler: () => void;
    /** Timer duration in seconds; 30 seconds by default */
    timer?: number;
}

const IndividualTrial: React.FC<IndividualTrialInterface> = ({timer = 30, ...props}) => {
    const [step, setStep] = useState<number>(0);
    const [points, setPoints] = useState<number>(0);
    const [isTimerDone, setIsTimerDone] = useState<boolean>(false);

    // Go to the next trial when the timer is done or the subject has done all the steps
    useEffect(() => {
        if (isTimerDone || step === 8) {
            props.onNextTrialHandler();
        }
    }, [step, isTimerDone]);

    const onNodeClickHandler = (currentNode: number, nextNode: number) => {
        // Update state
        setStep(step + 1);
        // Select current edge
        const currentEdge = props.edges.filter(
            (edge) => edge.source_num === currentNode && edge.target_num === nextNode)[0];
        // Update cumulative reward
        setPoints(points + currentEdge.reward);
    }

    return (
        <Paper sx={{p: 2, margin: 'auto', maxWidth: 700, flexGrow: 1}}>
            <Grid sx={{flexGrow: 1}} direction="row" container spacing={2}>
                {/* Network */}
                <Grid item>
                    <DynamicNetwork
                        nodes={props.nodes}
                        edges={props.edges}
                        onNodeClickParentHandler={onNodeClickHandler}
                        isDisabled={isTimerDone}
                    />
                </Grid>
                <Grid item sm container>
                    <Grid sx={{flexGrow: 1}} direction="column" container spacing={2}>
                        {/* Timer */}
                        <Box sx={{margin: "10px"}} justifyContent="center">
                            <Timer time={timer} OnTimeEndHandler={() => setIsTimerDone(true)}/>
                        </Box>
                        {/* Information */}
                        <Box sx={{p: 2, margin: "10px"}} justifyContent="center">
                            <Grid item>
                                <Typography variant="h5" component="div">
                                    Step {step}
                                </Typography>
                            </Grid>
                            <Grid item>
                                <Typography variant="h5" component="div">
                                    Points {points}
                                </Typography>
                            </Grid>
                        </Box>
                    </Grid>

                </Grid>
            </Grid>
        </Paper>
    );
};


export default IndividualTrial;