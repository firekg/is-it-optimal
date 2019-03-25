from util import normalize
from util import entropy
from concept import likelihood

import numpy as np
import matplotlib.pyplot as plt

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
    possible_y = np.array([0,1])
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


def simulate_performance(task, truth_ind=0, method="eig"):
    """
    level 0: a loop over scoring methods
    method can be "eig" or "self-teach"
    """
    concept_space = np.array(task)
    lik = likelihood(concept_space)
    n_concept, n_feature, n_y = lik.shape
    prior = np.array([1./n_concept]*n_concept)
    x_history = -np.ones(n_feature)
    y_history = -np.ones(n_feature)
    s_history = -np.ones(n_feature)
    performance = -np.ones(n_feature)
    for trial in range(n_feature):
        if method == "eig":
            score = expected_information_gain(lik, prior)
        elif method == "self-teach":
            score = self_teach(lik, prior)
        elif method == "random":
            score = random_selection(lik)
        elif method == "epg":
            score = expected_probability_gain(lik, prior)
        # forbid re-selection
        for ind in range(trial):
            score[int(x_history[ind])] = -np.inf
        # let argmax break ties on its own
        x = score.argmax()
        y = concept_space[truth_ind, x]
        post = posterior(int(x), int(y), lik, prior)
        x_history[trial] = x
        y_history[trial] = y
        s_history[trial] = score[int(x)]
        performance[trial] = post[truth_ind]
        prior = post
    return performance, x_history, y_history, s_history


def detailed_plots(task, method1, method2):
    n_concept, n_feature = task.shape
    x_plot = np.arange(n_feature)
    plt.figure(figsize=(20, 2))
    for i in range(n_concept):
        perf_m1, x_m1, _, s_history = simulate_performance(task, truth_ind=i, method=method1)
        if np.isnan(s_history).any():
            print("Warning: At least one probe is selected on NaN!")
        perf_m2, x_m2, _, s_history = simulate_performance(task, truth_ind=i, method=method2)
        if np.isnan(s_history).any():
            print("Warning: At least one probe is selected on NaN!")
        plt.subplot(1, n_concept, i+1)
        plt.plot(x_plot+1, perf_m1, '-ro')
        plt.plot(x_plot+1, perf_m2, '-bs')
        for j in x_plot:
            plt.text(j+1, perf_m1[j]+0.05, str(int(x_m1[j])), fontsize=8, color='r')
            plt.text(j+1, perf_m2[j]-0.13, str(int(x_m2[j])), fontsize=8, color='b')
        plt.xlim(0, n_feature+1)
        plt.ylim(0, 1.2)
        if i==0:
            plt.legend([method1, method2], loc='upper left', frameon=False)


def average_performance(task, method='eig'):
    """
    level 1: a loop over simulate performance
    """
    n_concept, n_feature = task.shape
    performance_mat = np.zeros([n_concept, n_feature])
    for i in range(n_concept):
        performance, _, _, s_history = simulate_performance(
                                        task, truth_ind=i, method=method)
        if np.isnan(s_history).any():
            print("Warning: At least one probe is selected on NaN!")
        performance_mat[i,:] = performance
    # the simple average corresponds to uniform prior
    return np.mean(performance_mat, axis=0)


if __name__ == '__main__':
    # test case 1: boundary task
    print("Boundary task:")
    boundary_task = [[1,1,1],
                     [1,1,0],
                     [1,0,0],
                     [0,0,0]]
    concept_space = np.array(boundary_task)
    lik = likelihood(concept_space)
    n_concept = len(concept_space)
    prior = np.array([1./n_concept]*n_concept)
    # EIG
    eig_score = expected_information_gain(lik, prior)
    print("EIG for boundary task calculated: {}".format(eig_score))
    answer = [np.log(4) - 3/4*np.log(3),
              np.log(4) - np.log(2),
              np.log(4) - 3/4*np.log(3)]
    print("EIG for boondary task should be: {}".format(answer))
    # Self-teaching
    st_score = self_teach(lik, prior)
    print("ST score for boundary task calculated: {}".format(st_score))
    answer = [25/77, 27/77, 25/77]
    print("ST for boondary task should be: {}".format(answer))


    # test case 2: line task
    print("Line task:")
    line_task = [[1,0,0],
                 [0,1,0],
                 [0,0,1],
                 [1,1,0],
                 [0,1,1],
                 [1,1,1]]
    concept_space = np.array(line_task)
    lik = likelihood(concept_space)
    n_concept = len(concept_space)
    prior = np.array([1./n_concept]*n_concept)
    # EIG
    eig_score = expected_information_gain(lik, prior)
    print("EIG for line task calculated: {}".format(eig_score))
    # Self-teaching
    st_score = self_teach(lik, prior)
    print("ST score for boundary task calculated: {}".format(st_score))


    # test case 3: Reproduce Table 1 row 1 of Nelson's paper
    # "Experience Matters: Information Acquisition Optimizes Probability Gain"
    # https://www.ncbi.nlm.nih.gov/pmc/articles/PMC2926803/pdf/nihms227105.pdf
    concept_space = np.array([[0, 0.57],[0.24, 0]])
    likelihood = np.zeros([2, 2, 2])
    likelihood[:,:,0] = concept_space
    likelihood[:,:,1] = 1 - concept_space
    prior = np.array([0.7, 0.3])
    score = expected_probability_gain(likelihood, prior)
    print("EPG calculated: {}".format(score))
    answer = [0.072, 0]
    print("EPG should be {}".format(answer))


    # identity task: random, eig, self-teach, prob-gain shoula all be the same
    # TODO: understand why random not the same as the others!
    task = np.identity(10)
    perf_eig = average_performance(task, method='eig')
    perf_st = average_performance(task, method='self-teach')
    perf_rand = average_performance(task, method='random')
    perf_epg = average_performance(task, method='epg')
