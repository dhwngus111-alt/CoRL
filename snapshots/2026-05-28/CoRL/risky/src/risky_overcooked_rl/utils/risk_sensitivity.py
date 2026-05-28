import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg
import torch
from numba import float32,boolean    # import the types
from numba.experimental import jitclass

spec = [
    ('b', float32),           # a simple scalar field
    ('lam', float32),
    ('eta_p', float32),
    ('eta_n', float32),
    ('delta_p', float32),
    ('delta_n', float32),
    ('mean_value_ref', boolean),
    ('is_rational', boolean),
    ('expected_td_targets', float32[:])
]

@jitclass(spec)
class CumulativeProspectTheory_Compiled:
    def __init__(self,b,lam,eta_p,eta_n,delta_p,delta_n,mean_value_ref = False):
        """
        Instantiates a CPT object that can be used to model human risk-sensitivity.
        :param b: reference point determining if outcome is gain or loss
        :param lam: loss-aversion parameter
        :param eta_p: exponential gain on positive outcomes
        :param eta_n: exponential loss on negative outcomes
        :param delta_p: probability weighting for positive outcomes
        :param delta_n: probability weighting for negative outcomes
        """
        # assert b==0, "Reference point must be 0"
        self.b = b
        self.lam = lam
        self.eta_p = eta_p
        self.eta_n = eta_n
        self.delta_p = delta_p
        self.delta_n = delta_n

        self.mean_value_ref = mean_value_ref
        if self.mean_value_ref:
            self.b = 0

        self.expected_td_targets = np.zeros(256, dtype=np.float32)

        self.is_rational = True
        if self.lam !=1: self.is_rational = False
        elif self.eta_p != 1: self.is_rational = False
        elif self.eta_n != 1: self.is_rational = False
        elif self.delta_p != 1: self.is_rational = False
        elif self.delta_n != 1: self.is_rational = False

    def expectation_samples(self,prospect_next_values, prospect_p_next_states,prospect_masks,reward,gamma):
        BATCH_SIZE = len(prospect_masks)
        self.expected_td_targets = np.zeros(BATCH_SIZE, dtype=np.float32)
        for i in range(BATCH_SIZE):
            prospect_mask = prospect_masks[i]
            prospect_values = prospect_next_values[prospect_mask, :]
            prospect_probs = prospect_p_next_states[prospect_mask, :]
            prospect_td_targets = reward[i, :] + (gamma) * prospect_values  # * (1 - done[i, :]) #(solving infinite horizon)

            if self.is_rational:
                self.expected_td_targets[i] = np.sum(prospect_td_targets.flatten() * prospect_probs.flatten())
            else:
                self.expected_td_targets[i] = self.expectation(prospect_td_targets.flatten(), prospect_probs.flatten())


        return self.expected_td_targets

    def expectation(self,values, p_values):
        """
        Applies the CPT-expectation multiple prospects (i.e. a series of value-probability pairs) which can arbitrarily
        replace the rational expectation operator E[v,p] = Σ(p*v). When dealing with more than two prospects, we must
        calculate the expectation over the cumulative probability distributions.
        :param values:
        :param p_values:
        :return:
        """
        if self.is_rational:
            # Rational Expectation
            return np.sum(values * p_values)
        if self.mean_value_ref:
            self.b = np.mean(values)

        # Step 1: arrange all samples in ascending order and get indexs of gains/losses
        sorted_idxs = np.argsort(values)
        sorted_v = values[sorted_idxs]
        sorted_p = p_values[sorted_idxs]

        K = len(sorted_v)  # number of samples
        if K == 1: return sorted_v[0]  # Single prospect = no CPT
        l = np.where(sorted_v <= self.b)[0]
        l = -1 if len(l) == 0 else l[-1]  # of no losses l=-1 indicator

        # Step 2: Calculate the cumulative liklihoods for gains and losses
        Fk = [min([max([0, np.sum(sorted_p[0:i + 1])]), 1]) for i in range(l + 1)] + \
             [min([max([0, np.sum(sorted_p[i:K])]), 1]) for i in range(l + 1, K)]  # cumulative probability
        Fk = Fk + [0]  # padding to make dealing with only gains or only losses easier

        # Step 3: Calculate biased expectation for gains and losses
        rho_p = self.perc_util_plus(sorted_v, Fk, l, K)
        rho_n = self.perc_util_neg(sorted_v, Fk, l, K)

        # Step 3: Add the cumulative expectation and return
        rho = rho_p - rho_n

        if self.mean_value_ref:
            rho += self.b
            self.b = 0
        return rho


    def perc_util_plus(self,sorted_v,Fk,l,K):
        """Calculates the cumulative expectation of all utilities percieved as gains"""
        rho_p = 0
        for i in range(l + 1, K):
            rho_p += self.u_plus(sorted_v[i]) * (self.w_plus(Fk[i]) - self.w_plus(Fk[i + 1]))
        # CLASSICAL FORMULATION ( no Fk =  Fk + [0]) -----------------------
        # for i in range(l + 1, K - 1):
        #     rho_p += self.u_plus(sorted_v[i]) * (self.w_plus(Fk[i]) - self.w_plus(Fk[i + 1]))
        # rho_p += self.u_plus(sorted_v[K - 1]) * self.w_plus(sorted_p[K - 1])
        return rho_p

    def perc_util_neg(self,sorted_v,Fk,l,K):
        """Calculates the cumulative expectation of all utilities percieved as losses"""
        # Fk =  Fk + [0]  # add buffer which results in commented out version below
        rho_n = 0
        for i in range(0, l + 1):
            rho_n += self.u_neg(sorted_v[i]) * (self.w_neg(Fk[i]) - self.w_neg(Fk[i - 1]))
        return rho_n
        # CLASSICAL FORMULATION ( no Fk =  Fk + [0]) -----------------------
        # rho_n = self.u_neg(sorted_v[0]) * self.w_neg(sorted_p[0])
        # for i in range(1, l + 1):
        #     rho_n += self.u_neg(sorted_v[i]) * (self.w_neg(Fk[i]) - self.w_neg(Fk[i - 1]))
        # return rho_n

    def u_plus(self,v):
        """ Weights the values (v) perceived as losses (v>b)"""
        return np.abs(v-self.b)**self.eta_p
    def u_neg(self, v):
        """ Weights the values (v) perceived as gains (v<=b)"""
        return self.lam * np.abs(v-self.b) ** self.eta_n
    def w_plus(self, p):
        """ Weights the probabilities p for probabilities of values perceived as gains  (v>b)"""
        delta = self.delta_p
        return p ** delta / ((p ** delta + (1 - p) ** delta) ** (1 / delta))
    def w_neg(self, p):
        """ Weights the probabilities p for probabilities of values perceived as losses (v<=b)"""
        delta = self.delta_n
        return p ** delta / ((p ** delta + (1 - p) ** delta) ** (1 / delta))

