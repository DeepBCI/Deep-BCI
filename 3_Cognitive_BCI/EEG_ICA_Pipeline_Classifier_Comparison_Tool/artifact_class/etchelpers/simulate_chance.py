import numpy as np

""" code for simulating chance levels in binary to multi class classification based on sample size
    Principle for code based on Mueller-Putz et al., 2008, IJB. Special thanks to L. Krol (TU-Berlin) for code insight."""

def simulate_chance(ntrials, alpha=0.05, nsims=25000, nclass=2):
    """

    :param ntrials: ndarray in numpy, length of the number of trials per class OR, sum of number of trials
    :param alpha: value, default 0.05
    :param nsims:
    :return: low confidence, mean, high confidence, # of correct samples required
    """

    if not np.shape(ntrials):
        ntrials = np.ones([1, nclass])*int(ntrials/nclass)
        ntrials = ntrials[0].astype(int)

    alpha *= 100
    total_trials = np.sum(ntrials)

    pvals = np.zeros(nsims)

    for simi in np.arange(nsims):
        x = []#np.asarray([])
        for classidx in np.arange(ntrials.shape[0]):
            #x = np.concatenate([x, np.ones([1,ntrials[classidx]])*classidx])
            x .append(np.ones([1, ntrials[classidx]]) * classidx)

        x = np.concatenate(x, axis=1).reshape(-1)
        y = np.random.choice(x.reshape(-1), x.shape[0], replace=False)

        pvals[simi] = np.sum(x == y) / total_trials

    # warning: np.percentile uses a different sampling meethod compared to matlab's prctile, which calls quantile.
    # todo: replicate matlab's quantile by adding constraints
    # return values: lower confidence interval, average simulated chance,
    # higher confidence interval, number of correct classifications required
    return [np.percentile(pvals, alpha/2), np.mean(pvals), np.percentile(pvals, 100-alpha/2), \
            np.ceil(np.percentile(pvals, 100-alpha/2)*total_trials)]
