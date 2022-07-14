# Copyright 2021 The NetKet Authors - All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import netket as nk
import jax.numpy as jnp

from optax._src import linear_algebra

def mycb(step, logged_data, driver):
    logged_data["acceptance"] = float(driver.state.sampler_state.acceptance)
    logged_data["globalnorm"] = float(linear_algebra.global_norm(driver._loss_grad))
    return True

def minimum_distance(x, sdim):
    """Computes distances between particles using mimimum image convention"""
    n_particles = x.shape[0] // sdim
    x = x.reshape(-1, sdim)

    distances = (-x[jnp.newaxis, :, :] + x[:, jnp.newaxis, :])[
        jnp.triu_indices(n_particles, 1)
    ]
    distances = jnp.remainder(distances + L / 2.0, L) - L / 2.0

    return jnp.linalg.norm(distances, axis=1)


def potential(x, sdim, eps, sigma):
    """Compute Gaussian potential for single sample x"""
    dis = minimum_distance(x, sdim)

    return eps * jnp.sum(jnp.exp(-dis**2/(2*sigma)))

import numpy as np
N = 32
d = 1/9.
eps = 1.
sigma = 1.
mass = 30.
L = np.sqrt(N)*3
print(L)
hilb = nk.hilbert.Particle(N=N, L=(L,L,), pbc=True)
sab = nk.sampler.MetropolisGaussian(hilb, sigma=0.05, n_chains=16, n_sweeps=32)

ekin = nk.operator.KineticEnergy(hilb, mass=mass)
pot = nk.operator.PotentialEnergy(hilb, lambda x: potential(x, 2, eps, sigma))
ha = ekin + pot

model = nk.models.DeepSetRelDistance(
    hilbert=hilb,
    cusp_exponent=0,
    layers_phi=2,
    layers_rho=3,
    features_phi=(16, 16),
    features_rho=(16, 16, 1),
)
vs = nk.vqs.MCState(sab, model, n_samples=4096, n_discard_per_chain=128)
vs.chunk_size = 1

import flax
with open("GC_32_2d.mpack", 'rb') as file:
    vs.variables = flax.serialization.from_bytes(vs.variables, file.read())

print(vs.samples.reshape(4096,2).shape)
import numpy as np

samples = vs.samples.reshape(4096,2)
print(vs.log_value(samples).shape)
samples = jnp.concatenate((samples, vs.log_value(samples)[:,jnp.newaxis]),axis=-1)
print(samples.shape)
np.save("samples.npy",samples)
"""
op = nk.optimizer.Sgd(0.05)
sr = nk.optimizer.SR(diag_shift=0.01)

gs = nk.VMC(ha, op, sab, variational_state=vs, preconditioner=sr)
gs.run(n_iter=1000, callback=mycb, out="GC_32_2d")
"""