"""Deterministic epoch-wise synthesis from the training pure pool only."""
import numpy as np
from rssnet_data import simulate


def synthesis_seed(seed, epoch):
    if seed < 0 or epoch < 0:
        raise ValueError("Seed and epoch must be nonnegative")
    # Domain-separated from the fixed validation/test seeds.
    return int(np.random.SeedSequence([seed, 0x52474550, epoch]).generate_state(1)[0])


class EpochMixtures:
    def __init__(self, pure_train, per_pair, seed, refresh=True):
        if per_pair < 1 or len(pure_train) != 7:
            raise ValueError("Expected seven nonempty training pools and positive per_pair")
        self.pure = pure_train
        self.per_pair, self.seed, self.refresh = per_pair, seed, refresh

    def generate(self, epoch):
        return simulate(self.pure, self.per_pair,
                        synthesis_seed(self.seed, epoch if self.refresh else 0))

    def protocol(self):
        return dict(mode="online_simulated" if self.refresh else "fixed_simulated",
                    per_pair=self.per_pair, mixtures_per_epoch=21*self.per_pair,
                    seed=self.seed, seed_domain="SeedSequence([seed, 0x52474550, epoch])",
                    refresh="every epoch" if self.refresh else "epoch zero reused",
                    source="manifest train partition ONLY; fixed train-median references",
                    generator="rssnet_data.simulate; same routine as validation",
                    alpha=[0.1, 0.9], snr_db=[10, 30],
                    noise="white + cumulative centered noise; nonnegative clipping",
                    background="quadratic, coefficient U(0,.08) times noisy maximum",
                    target="clean weighted endpoints / shared input L2 norm; not mass fractions")
