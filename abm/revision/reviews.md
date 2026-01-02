Reviewer #1 (Remarks to the Author):

This paper investigates the conditions under which machine-discovered strategies can be adopted and preserved within human cultural evolution. The authors combine a multi-generational social-learning experiment with agent-based simulations to test three hypothesised requirements for persistent “machine discoveries”: appropriate discovery difficulty, low transmission difficulty, and a recognizable selective advantage. In their laboratory task, deep-RL agents reliably discovered an “optimal” reward-network strategy that human participants, owing to their myopic loss-avoidance bias, almost never discovered themselves. Yet once this machine-generated strategy was introduced, it spread across human generations and remained detectable long after the machine players were gone. Simulations further explored the parameter space of discovery and transmission difficulty and confirmed that machine contributions can durably expand the human cultural repertoire. Overall, this is a very interesting and timely study, we do however have some points:

Major comments
1/
The experiment hinges on the idea that humans fail to discover the optimal strategy because of an inherent cognitive biases (such as prioritising immediate reward at the expense of long-term gains) whereas machines, unconstrained by such biases, can discover “alien” strategies. Yet humans were given five minutes to train on six networks, while the RL agents trained on 1,000 networks in the same time. Is human failure driven by a constitutive cognitive bias or simply by limited training opportunities? The authors should discuss or attempt to disentangle these two possibilities.
2/
Relatedly, what is the minimal number of reward network trials needed for the AI agent to reach an optimal policy? How many trials (or minutes) would human participants need to reliably discover the same strategy? If humans were given, say, 15 minutes, would machine discovery still look “alien,” or would the advantage disappear? Because the paper frames its contribution in analogy with Go and chess—where machines uncover strategies that humans have failed to find after centuries of play—this distinction is critical.
3/
In Experiment 1 (“Humans Preserve the Machine Discovery”), the authors report that in human-machine populations the optimal strategy persisted across generations and all final-generation players were descendants of a machine demonstrator. But the data mainly show that the reward-maximising strategy, whoever first discovers it, is preserved. The fact that machines find it more often seems due to their computational speed and the imposed time constraints, not to any uniquely “machine-like” quality of the strategy itself. This suggests that what is transmitted is simply the correct strategy, not necessarily a strategy with distinctive machine properties. Moreover, in 6 of 15 mixed populations the strategy eventually disappeared despite identical machine discovery processes. This pattern looks compatible with chance rather than any inherent “machine insight.” The authors should clarify whether they see evidence for a qualitatively distinct machine contribution beyond mere discovery frequency.
4/
The description of Figure 5 is difficult to follow. As presented, panel A appears to show a non-linear relationship in which higher transmission difficulty somehow predicts lower discovery difficulty. Please explain more clearly what the axes and boundaries represent and how to interpret the apparent non-linearity.
5/
Panel B is said to show the “boost in average reward of mixed populations over human-only populations in the final generation.” Presumably this heatmap represents human reward minus human-machine reward. Given the theoretical framing, one might expect the machine advantage to grow when discovery or transmission difficulty is high, i.e. precisely where machines should help most. Instead, the advantage appears bounded within moderate, human-achievable levels of difficulty. Could this indicate a limitation of the current simulation assumptions? Some discussion would be welcome.
6/
The three proposed conditions for machine discoveries to be culturally adopted—appropriate discovery difficulty, low transmission difficulty and recognizable selective advantage—are compelling, but their theoretical origin is not clearly explained. Please clarify why these conditions should be regarded as necessary (and perhaps sufficient) rather than merely plausible.
7/
The manuscript’s structure blends introduction, design and results in a less conventional way; if this is intentional or aligned with the journal’s style, discard this point
Minor comments

• When reporting regression results (of any type), please include p-values or at least a measure of model comparison or fit. Even if very large betas are almost certainly significant, it is preferable to document this explicitly.

• The simulation-based power analysis is only briefly described. Please expand on how it was performed, especially if pilot data informed the parameter choices.


• The procedure by which participants “recorded themselves” for future generations is not entirely clear. Did they write textual descriptions, use webcams, or something else? If written, did you examine whether surface features (e.g., spelling errors, verbosity) affected the likelihood that a strategy was preserved? How did the machine players produce their recorded demonstrations?

• The paper would benefit from a brief limitations section (with or without a dedicated header) to acknowledge, for example, the constrained laboratory setting and the potential confound between cognitive bias and training time.




Reviewer #2 (Remarks to the Author):

The article investigates the conditions under which machine-induced cultural shifts occur using controlled behavioral experiments and computer simulations. The study makes an important contribution by approaching the topic of cultural evolution with AI (simple reinforcement-learning agents). It is technically commendable that the authors conducted an experiment requiring about 40 participants per group (8 participants × 5 generations, in the human-population condition). I find this study meaningful.
While I have no concerns regarding the importance of the research question, there are several points the authors may wish to address to improve the manuscript regarding the analysis and interpretation that should be improved. Please find some comments and questions below. Because the document does not include line numbers, some of the following comments may be unclear regarding which parts are being discussed. I would appreciate it if the authors could include line numbers in the files when they have the opportunity to revise and resubmit the article.

1. There is a gap between what the authors state in the abstract and introduction and what they test in their experiment. The authors suggest that three conditions are involved in machine-induced cultural shift based on cultural evolution theory: Appropriate Discovery Difficulty, Low Transmission Difficulty, and Recognizable Selective Advantage. In the abstract, they state, “Using a cultural transmission experiment and an agent-based simulation, we demonstrate that when these conditions are met, machine-discovered strategies can be transmitted, understood, and preserved by human populations, leading to enduring cultural shifts.” However, these conditions are not manipulated in the experiment presented in the paper (the preregistration and the discussion mention pilot data, but from what I can see, it is not publicly available in the main paper). Of course, it makes sense to test the three conditions in a low-cost simulation and then test just one specific combination in a more expensive experiment. However, since a simulation and an experiment are not the same, what is demonstrated in the simulation has not been demonstrated in the experiment. Therefore, it may be inappropriate to state in the abstract and introduction that they have shown the importance of the three conditions using both the experiment “and” the simulation. Conversely, if they were to argue that the experiment and simulation are essentially the same, then the experiment would have been unnecessary, and the simulation alone would have sufficed.

2. In Figure 4, the dots represent individual populations (15 populations per condition), but the analysis reported in reference to Figure 4 seems to use each participant as the unit of analysis, which is misleading.

3. In Figure 4, the authors might want to include standard errors or CIs around the means.

4. In the preregistration, four models (1a, 1b, 2a, and 2b) were listed. The authors should clarify which of these models corresponds to the analysis they describe in the main text when they state, ”Our preregistered linear mixed effects model on these latter generations supported that this difference was significant.” Furthermore, if they used model 1a, they should explicitly state that it is a model that tests for an interaction. If they are reporting a model that does not include the interaction, they should also report the model that does include the interaction, as specified in their preregistration.

5. If individual participants are the unit of analysis, the model must explicitly account for the nesting of participants within trees and generations. Otherwise, there is a risk of a Type I error. Unfortunately, the preregistered models do not seem to include this hierarchical structure. I would recommend that the authors re-analyze their data with the appropriate nesting structures. If the model fails to converge due to its complexity, it suggests that the sample size was too small to begin with, and it is not a valid reason to ignore the hierarchy in the analysis.

6. In the discussion, the authors claim, “Our pilot experiments illustrated the key challenge of achieving the right balance between discovery difficulty and transmissibility.” If the authors wish to maintain this claim, they should report the pilot data publicly.



Reviewer #3 (Remarks to the Author):

This paper examines the conditions under which machines can become drivers of (persistent) cultural change. It studies cultural evolution in populations of humans trying to solve a problem, vs populations of humans and machines; and tries to determine what conditions are necessary for machine-discovered ideas to spread *and* be fully incorporated by humans. It identifies three such conditions for solutions: they must have a selective advantage, they must be non-trivial, and they must be learnable. This is done with cultural transmission experiments, that are nicely combined with computer simulations in order to examine counterfactuals and establish these conditions are all necessary.

For the experiment, a specific task is designed such that humans have a bias that makes it hard to discover optimal solutions. Machines, on the other hand, can quickly find optimal solutions using standard learning algorithms. Around ~1.2k participants were split into 30 populations; half were Human‑Only, and half were Human‑Machine, including some “machine players” inserted in Generation 0. After practicing solutions, participants chose a demonstrator from five candidates shown with their average scores (without knowing who was human or machine), watched that replay, reproduced it, and finally solved new networks and recorded their own demonstration for the next generation. This design allows the authors to test whether an machine‑discovered strategy would be learned by humans and preserved across generations even in the absence of machines.

The experiment is complemented with some agent-based simulations that mirror the lab setting. Here, each agent can pick a strategy (random, myopic, or optimal), and in each generation it either explored on its own or learned socially by picking a demonstrator from five candidates in the previous generation. The simulation allows control over discovery and transmission difficulty. The key result (Figure 5, p. 11) is that machines create a lasting boost only when the optimal strategy is hard for humans to discover but still easy to learn; if discovery is easy, learning is random, or transmission is too difficult, the gains provided by machine discoveries are small.

Some minor comments:

- I think it would be interesting to tease out more when each condition is required. The fact that selective advantage is necessary but not sufficient is somehow puzzling. I suppose this has to do not only with ideas spreading, but the fact that they need to be incorporated by humans in the absence of machines. If that is the case, would machines teaching each other will again guarantee that selective advantage on its own is enough? I feel this could be spelled out more clearly in the discussion.

- Can the authors explain the appeal of the random strategy in the simulation? Would the same results arise without it?

- Can the authors comment on what they would expect with tasks that are more fair or where the advantage of machines is large but not huge? The task is particularly easy to solve for machines: not even Q-learning would be required for an optimal solution to be found.


This is I think a great paper that combines a fantastic question with good methods. The topic is fascinating and will be of interest to computer scientists and social scientists alike. Overall, the paper convincingly shows that a machine‑discovered, non-trivial but learnable strategies can spread and persist. It would be good to extend these to other less engineered tasks where the advantage of machines is less salient, but this may be part of the many extensions I am sure the work will inspire.

I recommend publication.



Reviewer #3 (Remarks on code availability):

Code is complete and seems to be of high standard.


Reviewer #4 (Remarks to the Author):

This paper is a good, straightforward demonstration of what it sets out to do. The authors show that machines can discover solutions that are counterintuitive to humans, and that humans can adopt them. The experiment and the model seem solid, well executed, and clearly presented. I thought the fact that the agent-based model recovers the theoretical expectations presented in the introduction a nice touch. I don’t find the fact that humans can recognise good solutions and use them particularly surprising, and I don’t think it is my place to comment on whether the novelty of this result qualifies the paper for publication in Nature Communications. Below I summarise some concerns I have with the framing, and some suggestions for improving clarity.

One niggling question I have is how much the dynamic captured here is about machines specifically vs. humans being able to recognise and integrate new information in their strategies, and whether this difference even matters. If I read the results correctly (Fig. 3, S6), one human participant discovered the optimal strategy and passed it on to the next generation, but this took place at the end of the experiment (generations 3-4) so we don’t know if this strategy would have persisted further. The expectation is that it would have. Would the human participants have discovered the optimal solution more often with enough time (6 trials seems low)? The novel point is then not that machines can transform human culture, but that machines can generate new solutions that humans can readily use.

Another point that I think should be discussed directly in the paper is the role that payoff-bias (i.e. the fact that humans have access to score and can use score to dictate their social learning) plays for these results. I am assuming that participants chose the highest-scoring model most of the time (it would be nice to see some numbers on this), which again means humans have built-in mechanisms to take advantage of good information and good solutions. The phrasing in the results section is potentially misleading. The section titled “Humans Preserve the Machine Discovery” seems to imply there is something special about the machine solution that ensures it is preserved, and I suspect the something special is better score. I suggest the authors mention this directly in the text (or better yet, run additional analyses looking at the relationship between score and transmission). Similarly, the next section “Humans are Behaviourally Congruent with the Machines” seems to suggest the human participant intend to copy the machines – would you find the same congruence with the loss strategy discovered by the one human participant in the human-only condition?

Minor comments below:
• On page 7 please mention how many training trials generation 0 humans get, and mention how many individual learning trails generations 1-4 get and whether is it before/after social learning
• Page 9 – “Panel B in Figure 4 shows the proportion of human moves congruent with machine-generated moves, summarized by generation and population.” – which machine generated moves?
• Fig. 4c – why do human-only participants in generation 0 mention a loss strategy and how did this translate into the participants’ score?
• Figure 5a – I struggled to parse this description, consider adding more detail
• Page 12 – in the discussion on population size and complex solutions, I would argue that the kind of phenomenon described here might benefit smaller populations more. Larger populations might have a higher chance of discovering the optimal solution, which would make the machine intervention redundant. It all depends on how novel/outside the human range the machine-discovered solutions can be.
• Sentence “Our results apply particularly to scenarios in which human exploration is constrained by time rather than computational cost, machine exploration remains scalable” on page 14 might need editing
• The choices of parameter values for the agent-based model require justification beyond the fact that they nicely produced the curve in Fig. 2b – why the chosen d and t values, why was the discovery rate for machine agents set to 1000 higher than human agents, what does superior mean in “Upon discovering a new strategy, the agent updated its preferred strategy if the new strategy was superior to the previous one.”


Reviewer #5 (Remarks to the Author):

I co-reviewed this manuscript with one of the reviewers who provided the listed reports. This is part of the Nature Communications initiative to facilitate training in peer review and to provide appropriate recognition for Early Career Researchers who co-review manuscripts.

Reviewer #5 (Remarks on code availability):

I co-reviewed this manuscript with one of the reviewers who provided the listed reports. This is part of the Nature Communications initiative to facilitate training in peer review and to provide appropriate recognition for Early Career Researchers who co-review manuscripts
