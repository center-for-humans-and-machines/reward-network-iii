import NetworkTrial from "../NetworkTrial";
import React, {FC} from "react";
import useNetworkContext from "../../../contexts/NetworkContext";
import {Box, Button, Typography} from "@mui/material";
import LinearSolution from "../../Network/LinearSolution";


interface ITryYour {
    solution: number[];
    teacherId: number;
    teacherTotalScore: number;
    endTrial: (data: any) => void;
}

const TryYourself: FC<ITryYour> = ({solution, teacherId, teacherTotalScore, endTrial}) => {
    const {networkState} = useNetworkContext();

    return (
        <>
            {networkState.isNetworkFinished ? (

                <Box
                    sx={{width: '600px'}}
                    justifyContent="center"
                    alignItems="center"
                    style={{margin: 'auto', marginTop: '15%'}}
                >
                    <Typography variant="h6" gutterBottom align={'left'}>
                        Your solution total score: {networkState.points}
                    </Typography>
                    <LinearSolution
                        edges={networkState.network.edges}
                        nodes={networkState.network.nodes}
                        moves={networkState.moves}
                        showTutorial={networkState.tutorialOptions.linearSolution}
                        id={200}
                    />
                    <Typography variant="h6" gutterBottom align={'left'}>
                        Player {teacherId} total score: {teacherTotalScore}
                    </Typography>
                    <LinearSolution
                        edges={networkState.network.edges}
                        nodes={networkState.network.nodes}
                        moves={solution}
                        id={150}
                    />
                    <Box textAlign="center" marginTop="20px">
                        <Button onClick={() => endTrial({moves: networkState.moves})} variant="contained"
                                color="primary">Continue</Button>
                    </Box>

                </Box>
            ) : (
                <>
                    <Typography variant="h3" align='center'>
                        Try to collect at least as many points as player {teacherId}!
                    </Typography>
                    <NetworkTrial advisorTotalPoints={teacherTotalScore}/>
                </>
            )}
        </>
    );

}


export default TryYourself;