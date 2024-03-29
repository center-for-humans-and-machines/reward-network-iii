import React, { FC, useEffect } from "react";
import NetworkTrial from "../NetworkTrial";
import useNetworkContext from "../../../contexts/NetworkContext";
import { NETWORK_ACTIONS } from "../../../reducers/NetworkReducer";
import { Typography } from "@mui/material";

interface IObservation {
  solution: number[];
  teacherId: number;
  /** Delay in ms between each played move. Default is 2000ms. */
  delayBetweenMoves?: number;
  /** Start the animation from the parent component. Default is true. */
  playAnimation?: boolean;
  playerTotalPoints?: number;
}

const Observation: FC<IObservation> = (props) => {
  const { networkState, networkDispatcher } = useNetworkContext();
  const {
    solution,
    teacherId,
    playAnimation = true,
    delayBetweenMoves = 2000,
    playerTotalPoints = 0,
  } = props;

  useEffect(() => {
    console.log("Observation: useEffect: step", networkState.step);
    if (playAnimation) {
      setTimeout(() => {
        networkDispatcher({
          type: NETWORK_ACTIONS.NEXT_NODE,
          payload: {
            nodeIdx: solution[networkState.step + 1],
            // nextMove:
            //   networkState.step < solution.length
            //     ? solution[networkState.step + 2]
            //     : null,
          },
        });
      }, delayBetweenMoves);
    }
  }, [networkState.step, playAnimation]);

  return (
    <>
      <Typography variant="h5" align="center">
        Watch player {teacherId} solving the task
      </Typography>
      <NetworkTrial
        showComment={false}
        teacherId={teacherId}
        isTimerPaused={false}
        playerTotalPoints={playerTotalPoints}
        showTotalPoints={false}
        allowNodeClick={false}
      />
    </>
  );
};

export default Observation;
