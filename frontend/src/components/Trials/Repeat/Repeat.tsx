import NetworkTrial from "../NetworkTrial";
import React, {FC, useEffect} from "react";
import useNetworkContext from "../../../contexts/NetworkContext";
import {Typography} from "@mui/material";


interface IRepeat {
    solution: number[];

}

const Repeat: FC<IRepeat> = ({solution}) => {
    const {networkState, networkDispatcher} = useNetworkContext();

    useEffect(() => {


    }, [networkState.step]);


    return (
        <>
            <Typography variant="h3" align='center'>
                Repeat the solution by following the dashed line
            </Typography>
            <NetworkTrial/>
        </>
    );

}


export default Repeat;