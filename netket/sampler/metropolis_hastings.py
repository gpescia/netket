from .abstract_sampler import AbstractSampler
from ..stats import mean as _mean
from netket import random as _random

import math
import numpy as _np
from numba import jit, int64, float64
from .._jitclass import jitclass


class MetropolisHastings(AbstractSampler):
    """
    ``MetropolisHastings`` is a generic Metropolis-Hastings sampler using
    a transition kernel to perform moves in the Markov Chain.
    The transition kernel is used to generate
    a proposed state :math:`s^\prime`, starting from the current state :math:`s`.
    The move is accepted with probability

    .. math::
    A(s\rightarrow s^\prime) = \mathrm{min}\left (1,\frac{P(s^\prime)}{P(s)} F(e^{L(s,s^\prime)})\right),

    where the probability being sampled from is :math:`P(s)=|M(s)|^p. Here ::math::`M(s)` is a
    user-provided function (the machine), :math:`p` is also user-provided with default value :math:`p=2`,
    and :math:`L(s,s^\prime)` is a suitable correcting factor computed by the transition kernel.
    """

    def __init__(
        self, machine, kernel, n_chains=16, sweep_size=None, batch_size=None,
    ):
        """
        Constructs a new ``MetropolisHastings`` sampler given a machine and
        a kernel.

        Args:
            machine: A machine :math:`M(s)` used for the sampling.
                    The probability distribution being sampled
                    from is :math:`P(s)=|M(s)|^p`, where the power :math:`p`
                    is arbitrary, and by default :math:`p=2`.
            kernel: A kernel to generate random transitions from a given state as
                    well as uniform random states.
                    This must have two methods, `random_state` and `transition`.
                    `random_state` takes an input state (in batches) and
                    changes it in-place to a valid random state.
                    `transition` takes an input state (in batches) and
                    returns a batch of random states obtained transitioning from the initial state.
                    `transition` must also return an array containing the
                    `log_prob_corrections` :math:`L(s,s^\prime)`.
            n_chains: The number of Markov Chain to be run in parallel on a single process.
            sweep_size: The number of exchanges that compose a single sweep.
                    If None, sweep_size is equal to the number of degrees of freedom being sampled
                    (the size of the input vector s to the machine).
            batch_size: The batch size to be used when calling log_val on the given Machine.
                    If None, batch_size is equal to the number Markov chains (n_chains).

        """

        super().__init__(machine, n_chains)

        self.n_chains = n_chains

        self.sweep_size = sweep_size

        self._kernel = kernel

        self.machine_pow = 2.0
        self.reset(True)

    @property
    def n_chains(self):
        return self._n_chains

    @n_chains.setter
    def n_chains(self, n_chains):
        if n_chains < 0:
            raise ValueError("Expected a positive integer for n_chains ")

        self._n_chains = n_chains

        self._state = _np.zeros((n_chains, self._input_size))
        self._state1 = _np.copy(self._state)

        self._log_values = _np.zeros(n_chains, dtype=_np.complex128)
        self._log_values_1 = _np.zeros(n_chains, dtype=_np.complex128)
        self._log_prob_corr = _np.zeros(n_chains)

    @property
    def machine_pow(self):
        return self._machine_pow

    @machine_pow.setter
    def machine_pow(self, m_power):
        if not _np.isscalar(m_power):
            raise ValueError("machine_pow should be a scalar.")
        self._machine_pow = m_power

    @property
    def sweep_size(self):
        return self._sweep_size

    @sweep_size.setter
    def sweep_size(self, sweep_size):
        self._sweep_size = sweep_size if sweep_size != None else self._input_size
        if self._sweep_size < 0:
            raise ValueError("Expected a positive integer for sweep_size ")

    def reset(self, init_random=False):
        if init_random:
            self._kernel.random_state(self._state)
        self._log_values = self.machine.log_val(self._state, out=self._log_values)

        self._accepted_samples = 0
        self._total_samples = 0

    @staticmethod
    @jit(nopython=True)
    def acceptance_kernel(
        state, state1, log_values, log_values_1, log_prob_corr, machine_pow
    ):
        accepted = 0

        for i in range(state.shape[0]):
            prob = _np.exp(
                machine_pow * (log_values_1[i] - log_values[i] + log_prob_corr[i]).real
            )
            assert not math.isnan(prob)

            if prob > _random.uniform(0, 1):
                log_values[i] = log_values_1[i]
                state[i] = state1[i]
                accepted += 1

        return accepted

    def __next__(self):

        _log_val = self.machine.log_val
        _acc_kernel = self.acceptance_kernel
        _state = self._state
        _state1 = self._state1
        _log_values = self._log_values
        _log_values_1 = self._log_values_1
        _log_prob_corr = self._log_prob_corr
        _machine_pow = self._machine_pow
        _t_kernel = self._kernel.transition

        accepted = 0

        for sweep in range(self.sweep_size):

            # Propose a new state using the transition kernel
            _t_kernel(_state, _state1, _log_prob_corr)

            _log_values_1 = _log_val(_state1, out=_log_values_1)

            # Acceptance Kernel
            accepted += _acc_kernel(
                _state,
                _state1,
                _log_values,
                _log_values_1,
                _log_prob_corr,
                _machine_pow,
            )

        self._total_samples += self.sweep_size * self.n_chains
        self._accepted_samples += accepted

        return self._state

    @property
    def acceptance(self):
        """The measured acceptance probability."""
        return _mean(self._accepted_samples) / _mean(self._total_samples)
