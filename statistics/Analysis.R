#R version: 4.3.2
#packages used: lme4, ggplot2, Rmisc
#install.packages(c("lme4", "ggplot2", "Rmisc"))
library(lme4)
library(ggplot2)
library(Rmisc)

# -----------------------------------------------------------------------------
# Control which analysis sections to run (set to TRUE to run)
# -----------------------------------------------------------------------------
run_descriptive_figures <- FALSE   # Descriptive Figures (performance/individual plots)
run_descriptive_analysis <- FALSE  # Descriptive Analysis (loss strategy means)
run_performance <- FALSE           # Statistical models on Performance (h1a, h1b, h2a)
run_alignment <- TRUE              # Statistical models on Alignment (h1a, h1b, h2a)
run_strategies <- FALSE            # Written strategies (h2b)
use_random_categorical_generation <- FALSE  # If TRUE: use alternative specs for 1a and 2b (generation as categorical random effect)

# Bootstrap: set to FALSE to skip bootstrap CIs (faster run for testing)
run_bootstrap <- TRUE
# Number of bootstrap samples when run_bootstrap is TRUE (e.g. 100 for quick test, 1000 for final)
n_boot <- 1000L

# Parallel bootstrap: use multiple cores for confint(..., method = "boot")
# Set to 1 to disable parallelization
n_cores_boot <- max(1L, parallel::detectCores() - 1L)

# Output directory for text results (one subfolder per run with timestamp and optional name)
base_out_dir <- "statistics/output"
run_timestamp <- format(Sys.time(), "%Y%m%d_%H%M%S")
# Optional short label for this run (no spaces recommended); set "" to omit
run_name <- "alignment_fixed_generation"
run_id <- if (nzchar(run_name)) paste(run_timestamp, run_name, sep = "_") else run_timestamp
out_dir <- file.path(base_out_dir, run_id)
dir.create(out_dir, showWarnings = FALSE, recursive = TRUE)

##########################################
# Load data
##########################################

#file with individual moves per trial
moves <- read.csv("data/exp_processed/moves_w_alignment.csv")
#most important variables:
#condition: with or without AI
#generation: within a replication, from 0 to 3
#replication_idx: unique id for each replication
#solution_total_score: reward total per trial
#subject_id: unique per participant
#human_machine_match: judgment on move level, 1 if human and machine move match

#file with data on the player level
player <- read.csv("data/exp_processed/player.csv")

# file with coded player strategies
ratings_both <- read.csv("data/exp_strategies_coded/coded_strategies.csv")
#on (human) participant level, has the written strategies and our ratings for them
#loss_strategy codes 1 when strategy is present

##########################################
# Data preparation
##########################################

#aggregate moves to trial levels for performance
#points, only demonstration trials
demo <- subset(moves, moves$trial_type == "demonstration")
demo$branchID <- paste0(demo$replication_idx, demo$condition)
#this is the unique id for population

#different subsets for later
demo_agg <- subset(demo, demo$move_idx == 0)
demo_gen1plus <- subset(demo_agg, demo_agg$generation > 0)
demo_lastgen <- subset(demo_gen1plus, demo_gen1plus$generation == 4)
demo_firstgen <- subset(demo_gen1plus, demo_gen1plus$generation == 1)

# player ratings second strategy
player_ratings <- subset(ratings_both, ratings_both$written_strategy_idx == 1)

##########################################
# Descriptive Figures
##########################################
if (run_descriptive_figures) {
#some figures on performance
ci <- group.CI(solution_total_score ~ condition + generation, data = demo_agg)
ggplot(data = demo_agg, aes(x = generation, y = solution_total_score, color = condition)) + stat_summary(geom = "point", fun = mean) + geom_pointrange(aes(x = generation, y = solution_total_score.mean, ymin = solution_total_score.lower, ymax = solution_total_score.upper, color = condition), data = ci) + ylim(0,2700) + theme_light()
#w/o AI
ci <- group.CI(solution_total_score ~ condition + generation, data = subset(demo_agg, demo_agg$ai_player == "False"))
ggplot(data = subset(demo_agg, demo_agg$ai_player == "False"), aes(x = generation, y = solution_total_score, color = condition)) + stat_summary(geom = "point", fun = mean) + geom_pointrange(aes(x = generation, y = solution_total_score.mean, ymin = solution_total_score.lower, ymax = solution_total_score.upper, color = condition), data = ci) + ylim(0,2700) + theme_light()
runaverages <- aggregate(subset(demo_agg$solution_total_score, demo_agg$ai_player == "False"), by = list(subset(demo_agg$generation, demo_agg$ai_player == "False"), subset(demo_agg$replication_idx, demo_agg$ai_player == "False"), subset(demo_agg$condition, demo_agg$ai_player == "False")), FUN = mean)
ggplot(data = subset(demo_agg, demo_agg$ai_player == "False"), aes(x = generation, y = solution_total_score, color = condition)) + geom_point(aes(x = Group.1, y = x, color = Group.3), data = runaverages, size = 1, position= position_jitter(width = 0.07), alpha = 0.4) + stat_summary(geom = "point", fun = mean, size = 4, shape = 18) + coord_cartesian(ylim = c(1250,2500)) + xlab("Generation") + ylab("Average Reward") + scale_color_manual(values=c("orange", "blue"), labels=c("AI Tree", "Human Tree")) + theme_light() + theme(axis.text = element_text(size=12), axis.title = element_text(size = 14))
playeravg <- aggregate(subset(demo_agg$solution_total_score, demo_agg$ai_player == "False"), by = list(subset(demo_agg$generation, demo_agg$ai_player == "False"), subset(demo_agg$session_id, demo_agg$ai_player == "False"), subset(demo_agg$condition, demo_agg$ai_player == "False")), FUN = mean)
ggplot(data = subset(demo_agg, demo_agg$ai_player == "False"), aes(x = generation, y = solution_total_score, color = condition)) + geom_point(aes(x = Group.1, y = x, color = Group.3, group = Group.3), data = runaverages, size = 1.2, position= position_dodge(width = 0.4), alpha = 0.5) + stat_summary(aes(group = condition), geom = "point", fun = mean, size = 4, shape = 18, position = position_dodge(width = 0.4)) + coord_cartesian(ylim = c(1250,2500)) + xlab("Generation") + ylab("Average Reward") + scale_color_manual(values=c("orange", "blue"), labels=c("AI Tree", "Human Tree")) + theme_light() + theme(axis.text = element_text(size=12), axis.title = element_text(size = 14))
#gen0 excluding AI
ci <- group.CI(solution_total_score ~ condition + generation, data = subset(demo_agg, demo_agg$ai_player == "False"))
ggplot(data = subset(ci, ci$generation == 0), aes(x = condition, y = solution_total_score.mean, color = condition)) + geom_point() + geom_pointrange(aes(x = condition, y = solution_total_score.mean, ymin = solution_total_score.lower, ymax = solution_total_score.upper, color = condition)) + ylim(0,2700) + theme_light()
#max player score in gen0
ggplot(data = subset(player, player$generation == 0 & player$ai_player == "False"), aes(x = replication_idx, y = player_score, color = condition)) + stat_summary(geom = "point", fun = max) + ylim(0,2700) + theme_light()
gen0_mean_w_ai <- mean(subset(player$player_score, player$generation == 0 & player$ai_player == "False" & player$condition == "w_ai"))
gen0_mean_wo_ai <- mean(subset(player$player_score, player$generation == 0 & player$ai_player == "False" & player$condition == "wo_ai"))
writeLines(capture.output({
  cat("Mean player score gen0, w_ai:", gen0_mean_w_ai, "\n")
  cat("Mean player score gen0, wo_ai:", gen0_mean_wo_ai, "\n")
}), file.path(out_dir, "01_descriptive_gen0_means.txt"))
#individual improvement
individ <- subset(moves, moves$move_idx == 0)
individ_noAI <- subset(individ, individ$ai_player == "False")
individ_gen0 <- subset(individ_noAI, individ_noAI$generation == 0)
individ_nogen0 <- subset(individ, individ$generation > 0)
#plots
ci <- group.CI(solution_total_score ~ condition + trial_id, data = individ_gen0)
ggplot(data = individ_gen0, aes(x = trial_id, y = solution_total_score, color = condition)) + stat_summary(geom = "point", fun = mean) + geom_pointrange(aes(x = trial_id, y = solution_total_score.mean, ymin = solution_total_score.lower, ymax = solution_total_score.upper, color = condition), data = ci) + ylim(0,2700) + theme_light()
individ_nogen0 <- subset(individ_nogen0, individ_nogen0$trial_type %in% c("individual", "try_yourself", "demonstration"))
ci <- group.CI(solution_total_score ~ condition + trial_id, data = individ_nogen0)
ggplot(data = individ_nogen0, aes(x = trial_id, y = solution_total_score, color = condition)) + stat_summary(geom = "point", fun = mean) + geom_pointrange(aes(x = trial_id, y = solution_total_score.mean, ymin = solution_total_score.lower, ymax = solution_total_score.upper, color = condition), data = ci) + ylim(0,2700) + theme_light()
individ_noAI <- subset(individ_noAI, individ_noAI$trial_type %in% c("individual", "try_yourself", "demonstration"))
individ_noAI$gen <- ifelse(individ_noAI$generation > 0, "1-4", "0")
individ_noAI$trial_id <- ifelse(individ_noAI$trial_type == "demonstration" & individ_noAI$gen == "0", individ_noAI$trial_id + 10, individ_noAI$trial_id)
individ_noAI$trial_id <- ifelse(individ_noAI$trial_type == "individual" & individ_noAI$gen == "0" & individ_noAI$trial_id > 5, individ_noAI$trial_id * 1.5, individ_noAI$trial_id)
ggplot(data = individ_noAI, aes(x = trial_id, y = solution_total_score, color = condition, shape = gen)) + stat_summary(geom = "point", fun = mean) + theme_light()
}

##########################################
# Descriptive Analysis
##########################################
if (run_descriptive_analysis) {

# Q: Is the loss strategy present in the written strategies before social learning?

#change in written strats
#t1 is always 0, t2 is 1 or 2
#this includes the t1 strategies
ratings1to4 <- subset(ratings_both, ratings_both$generation > 0)
ratings1to4$written_strategy_idx <- ifelse(ratings1to4$written_strategy_idx == 2, 1, ratings1to4$written_strategy_idx)
ci <- group.CI(loss_strategy ~ condition + written_strategy_idx, data = ratings1to4)
ggplot(data = ratings1to4, aes(x = written_strategy_idx, y = loss_strategy, color = condition)) + stat_summary(geom = "point", fun = mean) + geom_pointrange(aes(x = written_strategy_idx, y = loss_strategy.mean, ymin = loss_strategy.lower, ymax = loss_strategy.upper, color = condition), data = ci) + theme_light()
loss_strat_1to4 <- unique(ave(ratings1to4$loss_strategy, ratings1to4$condition, ratings1to4$written_strategy_idx))

#include gen0
ratings_both$written_strategy_idx <- ifelse(ratings_both$written_strategy_idx == 2, 1, ratings_both$written_strategy_idx)
ratings_both$gen <- ifelse(ratings_both$generation > 0, "1-4", "0")
ratings_both$genchar <- as.character(ratings_both$generation)
ggplot(data = ratings_both, aes(x = written_strategy_idx, y = loss_strategy, color = condition, shape = gen)) + stat_summary(geom = "point", fun = mean) + theme_light()
ratings_gen0 <- subset(ratings_both, ratings_both$generation == 0)
loss_strat_gen0 <- unique(ave(ratings_gen0$loss_strategy, ratings_gen0$condition, ratings_gen0$written_strategy_idx))
writeLines(capture.output({
  cat("Loss strategy (ratings gen 1-4 by condition x written_strategy_idx):\n"); print(loss_strat_1to4)
  cat("\nLoss strategy (ratings gen 0 by condition x written_strategy_idx):\n"); print(loss_strat_gen0)
}), file.path(out_dir, "02_descriptive_loss_strategy_means.txt"))
}

##########################################
# Statistical Analysis
##########################################

#Statistical models on Performance
if (run_performance) {
#hyp 1a
ci <- group.CI(solution_total_score ~ condition + generation, data = demo_gen1plus)
ggplot(data = demo_gen1plus, aes(x = generation, y = solution_total_score, color = condition)) + stat_summary(geom = "point", fun = mean) + geom_pointrange(aes(x = generation, y = solution_total_score.mean, ymin = solution_total_score.lower, ymax = solution_total_score.upper, color = condition), data = ci) + ylim(0,2700) + theme_light()

demo_gen1plus$generation <- scale(demo_gen1plus$generation)
model_lm <- lm(solution_total_score ~ condition + generation, data = demo_gen1plus)
model_lmer1 <- lmer(solution_total_score ~ condition + generation + (1|session_id), data = demo_gen1plus)
model_lmer2 <- lmer(solution_total_score ~ condition * generation + (1|session_id) + (generation|branchID), data = demo_gen1plus)
if (run_bootstrap) { ci_perf_1a <- confint(model_lmer2, method = "boot", nsim = n_boot, verbose = TRUE, parallel = "multicore", ncpus = n_cores_boot) } else { ci_perf_1a <- NULL }
writeLines(capture.output({ print(summary(model_lm)); print(summary(model_lmer1)); print(summary(model_lmer2)); cat("\nBootstrap CI:\n"); if (run_bootstrap) print(ci_perf_1a) else cat("(skipped)\n") }), file.path(out_dir, "03_performance_h1a.txt"))

#hyp1b
ci2 <- group.CI(solution_total_score ~ condition, data = demo_lastgen)
ggplot(data = demo_lastgen, aes(x = condition, y = solution_total_score, color = condition)) + stat_summary(geom = "point", fun = mean) + geom_pointrange(aes(x = condition, y = solution_total_score.mean, ymin = solution_total_score.lower, ymax = solution_total_score.upper, color = condition), data = ci2) + ylim(0,3000) + theme_light()

model_lm <- lm(solution_total_score ~ condition, data = demo_lastgen)
model_lmer1 <- lmer(solution_total_score ~ condition + (1|session_id), data = demo_lastgen)
model_lmer2 <- lmer(solution_total_score ~ condition + (1|session_id) + (1|branchID), data = demo_lastgen)
if (run_bootstrap) { ci_perf_1b <- confint(model_lmer2, method = "boot", nsim = n_boot, verbose = TRUE, parallel = "multicore", ncpus = n_cores_boot) } else { ci_perf_1b <- NULL }
writeLines(capture.output({ print(summary(model_lm)); print(summary(model_lmer1)); print(summary(model_lmer2)); cat("\nBootstrap CI:\n"); if (run_bootstrap) print(ci_perf_1b) else cat("(skipped)\n") }), file.path(out_dir, "04_performance_h1b.txt"))

#hyp2a
ci2 <- group.CI(solution_total_score ~ condition, data = demo_firstgen)
ggplot(data = demo_firstgen, aes(x = condition, y = solution_total_score, color = condition)) + stat_summary(geom = "point", fun = mean) + geom_pointrange(aes(x = condition, y = solution_total_score.mean, ymin = solution_total_score.lower, ymax = solution_total_score.upper, color = condition), data = ci2) + ylim(0,3000) + theme_light()

model_lm <- lm(solution_total_score ~ condition, data = demo_firstgen)
model_lmer1 <- lmer(solution_total_score ~ condition + (1|session_id), data = demo_firstgen)
model_lmer2 <- lmer(solution_total_score ~ condition + (1|session_id) + (1|branchID), data = demo_firstgen)
if (run_bootstrap) { ci_perf_2a <- confint(model_lmer2, method = "boot", nsim = n_boot, parallel = "multicore", ncpus = n_cores_boot) } else { ci_perf_2a <- NULL }
#singular, fit all optimizers
model.all <- allFit(model_lmer2)
writeLines(capture.output({ print(summary(model_lm)); print(summary(model_lmer1)); print(summary(model_lmer2)); cat("\nBootstrap CI:\n"); if (run_bootstrap) print(ci_perf_2a) else cat("(skipped)\n"); cat("\nallFit (all optimizers):\n"); print(summary(model.all)) }), file.path(out_dir, "05_performance_h2a.txt"))
}

#stargazer(model, model2, model3, type = "text", column.labels = c("1a", "1b", "2a"), dep.var.caption =  "Prediction", dep.var.labels = "", digits = 1, model.numbers = FALSE, omit.stat = c("ll", "aic", "bic"), omit.table.layout = "n", report = "vcs")
#stargazer(model, model2, model3, column.labels = c("1a", "1b", "2a"), dep.var.caption =  "Prediction", dep.var.labels = "", digits = 1, model.numbers = FALSE, omit.stat = c("ll", "aic", "bic"), omit.table.layout = "n", report = "vcs")

#Statistical models on alignment
if (run_alignment) {
moves$human_machine_match <- ifelse(moves$human_machine_match == "True", 1, 0)
ci <- group.CI(human_machine_match ~ condition + generation, data = moves)
ggplot(data = moves, aes(x = generation, y = human_machine_match, color = condition)) + stat_summary(geom = "point", fun = mean) + geom_pointrange(aes(x = generation, y = human_machine_match.mean, ymin = human_machine_match.lower, ymax = human_machine_match.upper, color = condition), data = ci) + ylim(0,1) + theme_light()
#w/o AI
ci <- group.CI(human_machine_match ~ condition + generation, data = subset(moves, moves$ai_player == "False"))
ggplot(data = subset(moves, moves$ai_player == "False"), aes(x = generation, y = human_machine_match, color = condition)) + stat_summary(geom = "point", fun = mean) + geom_pointrange(aes(x = generation, y = human_machine_match.mean, ymin = human_machine_match.lower, ymax = human_machine_match.upper, color = condition), data = ci) + ylim(0,1) + theme_light()

aldemo <- subset(moves, moves$trial_type == "demonstration")
aldemo$branchID <- paste0(aldemo$replication_idx, aldemo$condition)
aldemo_gen1plus <- subset(aldemo, aldemo$generation > 0)
aldemo_lastgen <- subset(aldemo_gen1plus, aldemo_gen1plus$generation == 4)
aldemo_firstgen <- subset(aldemo_gen1plus, aldemo_gen1plus$generation == 1)

#hyp 1a
if (use_random_categorical_generation) {
  # Alternative specification: generation as categorical (fixed in glm, random in glmer)
  model_glm <- glm(human_machine_match ~ condition + factor(generation), data = aldemo_gen1plus, family = binomial(link = "logit"))
  model_glmer <- glmer(human_machine_match ~ condition + (1|session_id) + (1|generation) + (1|branchID), data = aldemo_gen1plus, family = binomial(link = "logit"))
} else {
  aldemo_gen1plus$generation <- scale(aldemo_gen1plus$generation)
  model_glm <- glm(human_machine_match ~ condition + generation, data = aldemo_gen1plus, family = binomial(link = "logit"))
  model_glmer <- glmer(human_machine_match ~ condition * generation + (1|session_id) + (generation|branchID), data = aldemo_gen1plus, family = binomial(link = "logit"))
}
#warning: this is a very large model and 1000 bootstraps will take several hours!
if (run_bootstrap) { ci_align_1a <- confint(model_glmer, method = "boot", nsim = n_boot, verbose = TRUE, parallel = "multicore", ncpus = n_cores_boot) } else { ci_align_1a <- NULL }
writeLines(capture.output({ print(summary(model_glm)); print(summary(model_glmer)); cat("\nBootstrap CI:\n"); if (run_bootstrap) print(ci_align_1a) else cat("(skipped)\n") }), file.path(out_dir, "06_alignment_h1a.txt"))

#hyp1b
model_glm <- glm(human_machine_match ~ condition, data = aldemo_lastgen, family = binomial(link = "logit"))
model_glmer <- glmer(human_machine_match ~ condition + (1|session_id) + (1|branchID), data = aldemo_lastgen, family = binomial(link = "logit"))
if (run_bootstrap) { ci_align_1b <- confint(model_glmer, method = "boot", nsim = n_boot, verbose = TRUE, parallel = "multicore", ncpus = n_cores_boot) } else { ci_align_1b <- NULL }
writeLines(capture.output({ print(summary(model_glm)); print(summary(model_glmer)); cat("\nBootstrap CI:\n"); if (run_bootstrap) print(ci_align_1b) else cat("(skipped)\n") }), file.path(out_dir, "07_alignment_h1b.txt"))

#hyp2a
model_glm <- glm(human_machine_match ~ condition, data = aldemo_firstgen, family = binomial(link = "logit"))
model_glmer <- glmer(human_machine_match ~ condition + (1|session_id) + (1|branchID), data = aldemo_firstgen, family = binomial(link = "logit"))
if (run_bootstrap) { ci_align_2a <- confint(model_glmer, method = "boot", nsim = n_boot, verbose = TRUE, parallel = "multicore", ncpus = n_cores_boot) } else { ci_align_2a <- NULL }
model.all <- allFit(model_glmer)
writeLines(capture.output({ print(summary(model_glm)); print(summary(model_glmer)); cat("\nBootstrap CI:\n"); if (run_bootstrap) print(ci_align_2a) else cat("(skipped)\n"); cat("\nallFit (all optimizers):\n"); print(summary(model.all)) }), file.path(out_dir, "08_alignment_h2a.txt"))
}

#stargazer(model, model2, model3, type = "text", column.labels = c("1a", "1b", "2a"), dep.var.caption =  "Prediction", dep.var.labels = "", digits = 3, model.numbers = FALSE, omit.stat = c("ll", "aic", "bic"), omit.table.layout = "n", report = "vcs")
#stargazer(model, model2, model3, column.labels = c("1a", "1b", "2a"), dep.var.caption =  "Prediction", dep.var.labels = "", digits = 3, model.numbers = FALSE, omit.stat = c("ll", "aic", "bic"), omit.table.layout = "n", report = "vcs")

#written strategies
if (run_strategies) {

#hyp2b
player_ratings$branchID <- paste0(player_ratings$replication_idx, player_ratings$condition)
ratings_gen1plus <- subset(player_ratings, player_ratings$generation > 0)

ci <- group.CI(loss_strategy ~ condition + generation, data = player_ratings)
ggplot(data = player_ratings, aes(x = generation, y = loss_strategy, color = condition)) + stat_summary(geom = "point", fun = mean) + geom_pointrange(aes(x = generation, y = loss_strategy.mean, ymin = loss_strategy.lower, ymax = loss_strategy.upper, color = condition), data = ci) + ylim(0,1) + theme_light()
runaverages <- aggregate(player_ratings, by = list(player_ratings$generation, player_ratings$replication_idx, player_ratings$condition), FUN = mean)
ggplot(data = player_ratings, aes(x = generation, y = loss_strategy, color = condition)) + geom_point(aes(x = Group.1, y = loss_strategy, color = Group.3, group = Group.3), data = runaverages, size = 1.2, position= position_dodge(width = 0.4), alpha = 0.5) + stat_summary(aes(group = condition), geom = "point", fun = mean, size = 4, shape = 18, position = position_dodge(width = 0.4)) + coord_cartesian(ylim = c(0,1)) + xlab("Generation") + ylab("Loss Strategy") + scale_color_manual(values=c("orange", "blue"), labels=c("AI Tree", "Human Tree")) + theme_light() + theme(axis.text = element_text(size=12), axis.title = element_text(size = 14))

if (use_random_categorical_generation) {
  # Alternative specification: generation as categorical (fixed in glm, random in glmer)
  model_glm <- glm(loss_strategy ~ condition + factor(generation), data = ratings_gen1plus, family = binomial(link = "logit"))
  model_glmer <- glmer(loss_strategy ~ condition + (1|generation) + (1|branchID), data = ratings_gen1plus, family = binomial(link = "logit"))
} else {
  ratings_gen1plus$generation <- scale(ratings_gen1plus$generation)
  model_glm <- glm(loss_strategy ~ condition + generation, data = ratings_gen1plus, family = binomial(link = "logit"))
  model_glmer <- glmer(loss_strategy ~ condition * generation + (generation|branchID), data = ratings_gen1plus, family = binomial(link = "logit"))
}
if (run_bootstrap) { ci_strat_2b <- confint(model_glmer, method = "boot", nsim = n_boot, verbose = TRUE, parallel = "multicore", ncpus = n_cores_boot) } else { ci_strat_2b <- NULL }
model.all <- allFit(model_glmer)
writeLines(capture.output({ print(summary(model_glm)); print(summary(model_glmer)); cat("\nBootstrap CI:\n"); if (run_bootstrap) print(ci_strat_2b) else cat("(skipped)\n"); cat("\nallFit (all optimizers):\n"); print(summary(model.all)) }), file.path(out_dir, "09_strategies_h2b.txt"))
}

#stargazer(model, type = "text", column.labels = "2b", dep.var.caption =  "Prediction", dep.var.labels = "", digits = 3, model.numbers = FALSE, omit.stat = c("ll", "aic", "bic"), omit.table.layout = "n", report = "vcs")
#stargazer(model, column.labels = "2b", dep.var.caption =  "Prediction", dep.var.labels = "", digits = 3, model.numbers = FALSE, omit.stat = c("ll", "aic", "bic"), omit.table.layout = "n", report = "vcs")
