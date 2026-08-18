# πBO plugin (prior slot)

Compile a belief from context into a factorized distribution, then fold it into the acquisition: α_π(x) = α(x) * π(x)^{β/(t+1)}. The exponent decays so a wrong prior fades as data arrives.

```bash
lenz set-belief --state ./state.json --prior '{
  "lr": {"dist": "lognormal", "mu": -7.0, "sigma": 1.0},
  "dropout": {"dist": "beta", "a": 2, "b": 8}
}' --decay-beta 10
```

Supported `dist` values: `uniform`, `normal` (mu, sigma), `lognormal` (mu, sigma), `beta` (a, b on the range scaled to [0,1]), `categorical` (`probs` map). Name a numeric distribution. Do not pass adjectives such as "aggressive".

```bash
lenz set-belief --state ./state.json --prior '{}' --clear
```

Cite: Hvarfner et al., 2022.
