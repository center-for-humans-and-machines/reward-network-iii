// NOTE: id, title, text and tip should be non-empty strings
export const tutorialTooltipContent = [
    {
        id: "practice_node",
        title: "Circle",
        text: "You start at the highlighted circle in the network.",
        tip: "Click a node",
    },
    {
        id: "practice_edge",
        title: "Arrow",
        text: "Circles are connected by different arrows. Your goal is to select a path along the arrows to earn points. You earn or lose points depending on the color of the arrow. Now click on the next node.",
        tip: "Click a node",
    },
    {
        id: "practice_step_score",
        title: "Score & Step",
        text: "You always have 8 moves per network. Your goal is to collect the maximum total number of points in these 8 moves. Now click on the next node.",
        tip: "Current step and cumulative score",
    },
    {
        id: "practice_linear_solution",
        title: "Your Solution",
        text: "As you proceed in the network, your progress is noted here. Now finish the trial by making your 8 moves.",
        tip: "Your Solution",
    },
    {
        id: "practice_timer",
        title: "Time Constraint",
        text: "In the actual experiment, you will have a limited time to solve each network. If you run out of time, you will receive -140 points for each move you are missing.",
        tip: "Time Constraint",
    },
    {
        id: "practice_total_score",
        title: "Total Score",
        text: "Your total number of points collected are displayed here. The points will determine your bonus payment. Note that some rounds, just like this, are only for practice. You will not to collect points in these rounds.",
        tip: "Total Score",

    },
    {
        id: "social_learning_selection_player",
        title: "Player Selection",
        text: `Select a player to see the solution of this player`,
        tip: "Select a player",
    },
    {
        id: "social_learning_observation_comment",
        title: "Player Comment",
        text: "The player might have provided their strategy for you here (or not, if the box is empty).",
        tip: "Player comment",
    },
    {
        id: "social_learning_observation_animation",
        title: "Player Solution Animation",
        text: "You can now watch their chosen path once. The animation will start in 4 seconds.",
        tip: "Player solution animation",
    },
];