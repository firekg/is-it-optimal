import numpy as np


class GraphActiveLearner:
    def __init__(self, graphs):
        self.hyp = graphs
        self.n_hyp = len(graphs)

        self.actions = np.array([1, 2, 3])
        self.n_actions = len(self.actions)

        # the set of possible observations
        # 0 = intervene, 1 = observed_off, 2 = observed_on
        self.observations = np.array([[0, 1, 1], [0, 1, 2],
                                      [0, 2, 1], [0, 2, 2],
                                      [1, 0, 1], [1, 0, 2],
                                      [1, 1, 0], [1, 2, 0],
                                      [2, 0, 1], [2, 0, 2],
                                      [2, 1, 0], [2, 2, 0]])
        self.n_observations = len(self.observations)

        # the set of possible interventions
        self.interventions = np.array([0, 0, 0, 0,
                                       1, 1, 2, 2,
                                       1, 1, 2, 2])
        self.n_interventions = len(np.unique(self.interventions))

        # prior over graphs
        self.prior = 1 / self.n_hyp * \
            np.ones((self.n_hyp, self.n_observations))

        assert np.allclose(np.sum(self.prior, axis=0), 1.0)

    def likelihood(self):
        """Calculate p(d|h, i)"""

        lik = np.zeros((self.n_hyp,
                        self.n_observations))

        for i, h in enumerate(self.hyp):
            lik[i] = h.likelihood()

        # the likelihood should sum to 3.0
        assert np.allclose(np.sum(lik, axis=1), 3.0)

        return lik

    # added by SY
    def reset_prior(self, prob):
        if prob.shape == self.prior.shape:
            self.prior = prob
        else:
            raise ValueError("prob input shape does not match self.prior")

    def update_posterior(self):
        """Calculates the posterior over all possible action/observation pairs
        for each graph"""
        self.posterior = self.likelihood() * self.prior
        denom = np.sum(self.posterior, axis=0)

        self.posterior = np.nan_to_num(np.divide(
            self.posterior, denom, where=denom != 0))

        # check sum of posterior is either 0s or 1s
        assert np.all(np.logical_or(
            np.isclose(np.sum(self.posterior, axis=0), 1.0),
            np.isclose(np.sum(self.posterior, axis=0), 0.0)))

    def prior_entropy(self):
        prior_entropy = np.nansum(self.prior * np.log2(1/self.prior), axis=0)

        # only consider unique interventions
        unique_interventions = np.array([0, 4, 6])
        prior_entropy = prior_entropy[unique_interventions]

        return prior_entropy

    def posterior_entropy(self):
        inv_posterior = np.divide(1, self.posterior, where=self.posterior != 0)
        log_inv_posterior = np.log2(inv_posterior, where=inv_posterior != 0)
        posterior_entropy = np.nansum(
            self.posterior * log_inv_posterior, axis=0)

        return posterior_entropy

    def observation_likelihood(self):
        obs_lik = np.sum(self.prior * self.likelihood(), axis=0)

        assert np.array_equal(self.posterior, np.nan_to_num(
            np.divide((self.likelihood() * self.prior), obs_lik, where=obs_lik != 0)))

        return obs_lik

    def expected_information_gain(self):
        weighted_posterior_entropy = np.zeros(self.n_actions)

        joint_posterior_entropy = self.observation_likelihood() * \
            self.posterior_entropy()

        # sum over possible observations
        for i in range(self.n_actions):
            weighted_posterior_entropy[i] = np.sum(
                joint_posterior_entropy[self.interventions == i])

        eig = self.prior_entropy() - weighted_posterior_entropy
        eig = eig / np.sum(eig)
        return eig
