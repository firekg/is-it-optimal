from util import normalize
from util import entropy
from concept import likelihood
from graph_active_learner import GraphActiveLearner

import numpy as np
import matplotlib.pyplot as plt

# changes made relative to active_learning.py:
# full_posterior()
# simulate_performance()
# detailed_plots()
# average_performance()
# add a function: wk2sy_full_prob(wk_prob)
# after if __name__ == '__main__':


def posterior(xs, ys, likelihood, prior):
    """
    xs = features Selected
    ys = values of x
    likelihood has form: n_concept, n_feature, n_y
    prior has form: n_concept
    posterior has form: n_concept
    P(h|x,y) = P(y|h,x)P(h) / sum_h P(y|h,x)P(h)
    """
    n_concept, _, _ = likelihood.shape
    posterior = np.zeros(n_concept)
    for i in range(n_concept):
        posterior[i] = np.product(likelihood[i, xs, ys])*prior[i]
    return normalize(posterior)


def full_posterior(likelihood, prior):
    """
    obtain posterior for all possible (x,y)
    """
    n_concept, n_feature, n_y = likelihood.shape
    full_posterior = np.zeros_like(likelihood)
    possible_x = np.arange(n_feature)
    possible_y = np.arange(n_y)
    for ind_x, x in enumerate(possible_x):
        for ind_y, y in enumerate(possible_y):
            full_posterior[:,ind_x, ind_y] = posterior(x, y, likelihood, prior)
    return full_posterior


def predictive(likelihood, belief):
    """
    likelihood has form: n_concept, n_feature, n_y
    belief has form: n_concept
    predictive has form: n_feature, n_y
    P(y*|x*,x,y) = sum_h P(y*|h,x*)P(h|x,y)
    """
    n_concept, n_feature, n_y = likelihood.shape
    rep_belief = full_shape_belief(belief, likelihood)
    return np.sum(likelihood*rep_belief, axis=0)


def full_shape_belief(belief, full_shape_array):
    """
    replicate belief to the full shape, the shape of likelihood
    """
    rep_belief = np.zeros_like(full_shape_array)
    for ind, val in enumerate(belief):
        rep_belief[ind,:,:] = val
    return rep_belief


def expected_information_gain(likelihood, prior):
    """
    expected_post_entropy has shape n_feature
    """
    n_concept, n_feature, n_y = likelihood.shape
    full_post = full_posterior(likelihood, prior)
    prior_predictive = predictive(likelihood, prior)
    full_post_entropy = np.zeros([n_feature, n_y])
    for ind_x in range(n_feature):
        for ind_y in range(n_y):
            full_post_entropy[ind_x, ind_y] = entropy(full_post[:,ind_x,ind_y])
    expected_post_entropy = np.sum(full_post_entropy*prior_predictive, axis=1)
    prior_entropy = entropy(prior)
    return prior_entropy - expected_post_entropy


def expected_probability_gain(likelihood, prior):
    n_concept, n_feature, n_y = likelihood.shape
    prior_max = prior.max()
    full_post = full_posterior(likelihood, prior)
    full_post_max = full_post.max(axis=0)
    full_prob_gain = full_post_max - prior_max
    prior_predictive = predictive(likelihood, prior)
    expected_prob_gain = np.sum(full_prob_gain*prior_predictive, axis=1)
    return expected_prob_gain


def self_teach(likelihood, prior):
    n_concept, n_feature, n_y = likelihood.shape
    full_post = full_posterior(likelihood, prior)
    # teacher's prior--this is probably not needed, as it should cancel out
    teacher_prior = np.zeros_like(likelihood)
    teacher_prior = 1./n_feature/n_y
    # normalizing constant--distinctiveness^-1
    z = np.sum(np.sum(full_post*teacher_prior, axis=2), axis=1)
    rep_z = full_shape_belief(z, full_post)
    rep_prior = full_shape_belief(prior, full_post)
    # compute self-teaching score
    # rep_z could be 0 and result in 0/0
    score = full_post*teacher_prior*rep_prior/rep_z
    score = np.nansum(np.nansum(score, axis=2), axis=0)
    # score should sum to 1
    if not np.isclose(np.sum(score), 1):
        print("Warning: Self-teaching score should sum to 1 but does not.")
    return score


def random_selection(likelihood):
    n_concept, n_feature, n_y = likelihood.shape
    return np.random.random(n_feature)


# TODO: add Gureckis' confirmation bias


def simulate_performance(task, truth_ind=0, method="eig", n_step=3):
    """
    level 0: a loop over scoring methods
    method can be "eig", "epg", "self-teach"
    """
    concept_space = np.array(task)
    # from task to likelihood
    ag = GraphActiveLearner(task) # taken from wk' code
    lik = wk2sy_full_prob(ag.likelihood()) # reformat
    n_concept, n_feature, n_y = lik.shape
    prior = np.array([1./n_concept]*n_concept)
    x_history = -np.ones(n_step)
    y_history = -np.ones(n_step)
    s_history = -np.ones(n_step)
    performance = -np.ones(n_step)
    for trial in range(n_step):
        if method == "eig":
            score = expected_information_gain(lik, prior)
        elif method == "self-teach":
            score = self_teach(lik, prior)
        elif method == "random":
            score = random_selection(lik)
        elif method == "epg":
            score = expected_probability_gain(lik, prior)
        # -----------------------------------------
        # deterministic sampling of intervention and observation
        # forbid re-selection
        # for ind in range(trial):
        #     score[int(x_history[ind])] = -np.inf
        # let argmax break ties on its own
        # x = score.argmax()
        # y = obs_lik.argmax()
        # -----------------------------------------
        # probabilistic sampling of intervention and observation
        # need to deal with numerical errors and edge cases...
        score = np.abs(score)
        if np.isclose(np.sum(score), 0):
            score = np.ones(n_feature)
        score = normalize(score)
        # print(score)
        x = np.random.choice(n_feature, 1, p=score)
        # print(x)
        obs_lik = normalize(lik[truth_ind, x, :])
        # print(obs_lik)
        y = np.random.choice(n_y, 1, p=obs_lik[0])
        # -----------------------------------------
        post = posterior(int(x), int(y), lik, prior)
        x_history[trial] = x
        y_history[trial] = y
        s_history[trial] = score[int(x)]
        performance[trial] = post[truth_ind]
        prior = post
    return performance, x_history, y_history, s_history


def detailed_plots(task, n_step=3, method1='eig', method2='self-teach'):
    ag = GraphActiveLearner(task)
    lik = wk2sy_full_prob(ag.likelihood())
    n_concept, n_feature, n_y = lik.shape
    x_plot = np.arange(n_step)
    plt.figure(figsize=(20, 2))
    for i in range(n_concept):
        perf_m1, x_m1, _, s_history = simulate_performance(task, truth_ind=i,
                                                method=method1, n_step=n_step)
        if np.isnan(s_history).any():
            print("Warning: At least one probe is selected on NaN!")
        perf_m2, x_m2, _, s_history = simulate_performance(task, truth_ind=i,
                                                method=method2, n_step=n_step)
        if np.isnan(s_history).any():
            print("Warning: At least one probe is selected on NaN!")
        plt.subplot(1, n_concept, i+1)
        plt.plot(x_plot+1, perf_m1, '-ro')
        plt.plot(x_plot+1, perf_m2, '-bs')
        for j in x_plot:
            plt.text(j+1, perf_m1[j]+0.05, str(int(x_m1[j])),
                                                    fontsize=8, color='r')
            plt.text(j+1, perf_m2[j]-0.13, str(int(x_m2[j])),
                                                    fontsize=8, color='b')
        plt.xlim(0, n_step+1)
        plt.ylim(0, 1.2)
        if i==0:
            plt.legend([method1, method2], loc='upper left', frameon=False)


def average_performance(task, method='eig', n_step=3):
    """
    level 1: a loop over simulate performance
    """
    ag = GraphActiveLearner(task)
    lik = wk2sy_full_prob(ag.likelihood())
    n_concept, n_feature, n_y = lik.shape
    performance_mat = np.zeros([n_concept, n_step])
    for i in range(n_concept):
        performance, _, _, s_history = simulate_performance(task, truth_ind=i,
                                                method=method, n_step=n_step)
        if np.isnan(s_history).any():
            print("Warning: At least one probe is selected on NaN!")
        performance_mat[i,:] = performance
    # the simple average corresponds to uniform prior
    return np.mean(performance_mat, axis=0)


def wk2sy_full_prob(wk_prob):
    """
    transform Wai Keen's likelihood format to my likelihood format
    """
    n_concept = len(wk_prob)
    n_feature = 3
    n_obs = 8   # 2**n_feature
    sy_prob = np.zeros([n_concept, n_feature, n_obs])
    for i, concept in enumerate(wk_prob):
        sy_prob[i,0,4] = wk_prob[i,0]
        sy_prob[i,0,5] = wk_prob[i,1]
        sy_prob[i,0,6] = wk_prob[i,2]
        sy_prob[i,0,7] = wk_prob[i,3]
        sy_prob[i,1,2] = wk_prob[i,4]
        sy_prob[i,1,3] = wk_prob[i,5]
        sy_prob[i,2,1] = wk_prob[i,6]
        sy_prob[i,2,3] = wk_prob[i,7]
        sy_prob[i,1,6] = wk_prob[i,8]
        sy_prob[i,1,7] = wk_prob[i,9]
        sy_prob[i,2,5] = wk_prob[i,10]
        sy_prob[i,2,7] = wk_prob[i,11]
    return sy_prob


if __name__ == '__main__':
    from dag import DirectedGraph
    from graph_utils import create_graph_hyp_space
    from graph_utils import create_active_learning_hyp_space
    from graph_active_learner import GraphActiveLearner
    from graph_self_teacher import GraphSelfTeacher

    active_graph_space = create_active_learning_hyp_space(t=0.8, b=0.0)

    # Check that WK's and SY's EIG is the same
    for space in active_graph_space:
        ag = GraphActiveLearner(space)
        ag.update_posterior()
        print("WK's EIG: {}".format(ag.expected_information_gain()))
        lik = wk2sy_full_prob(ag.likelihood())
        # print(lik.shape)
        prior = np.array([0.5, 0.5])
        eig = expected_information_gain(lik, prior)
        print("SY's EIG: {}".format(normalize(eig)))
        print("")

    # Check that WK's and SY's self-teaching is the same
    for space in active_graph_space:
        stg = GraphSelfTeacher(space)
        stg.update_learner_posterior()
        print("WK's self-teaching score: {}".format(stg.update_self_teaching_posterior()))
        lik = wk2sy_full_prob(stg.likelihood())
        # print(lik.shape)
        prior = np.array([0.5, 0.5])
        st_score = self_teach(lik, prior)
        print("SY's self-teaching score: {}".format(st_score))
        print("")
