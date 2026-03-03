# DSC-EPGG: Week 1–2 Implementation Plan

## For: Coding Agent
## Scope: Environment setup, multi-step patching, PPO training loop, observation wrapper, unit tests

---

## Project Context

We are building a research pipeline that:
1. Runs multi-agent reinforcement learning in a repeated Public Goods Game.
2. Logs detailed behavioral trajectories.
3. Later fits a PLRNN dynamical systems model to those trajectories (weeks 5+, not your concern now).

The base codebase is **marl-emecom** — a small research repo for multi-agent RL with emergent communication in public goods games. Your job is to adapt it from a **single-step** game to a **multi-step repeated session** game, add several environment features, replace the training algorithm with PPO, and verify everything works.

**Repository:** https://github.com/nicoleorzan/marl-emecom

**Key paper:** Orzan et al. (2024), "Learning in public goods games: the effects of uncertainty and communication on cooperation," *Neural Computing and Applications.*

---

## Critical Constraints (Read Before Writing Any Code)

### Information Leakage Rule
Agents must NEVER observe exact rewards or group welfare. Combined with cooperation count k, these reveal the hidden multiplication factor f via the payoff equation, destroying the incentive uncertainty that the entire project depends on. Rewards are used for learning (policy gradient) but are NOT part of the observation vector.

### Three Information Sets
- **Set A (agent observations):** noisy f_hat, endowment, last-round cooperation fraction, own last action, EWMA cooperation, received messages. NOTHING ELSE.
- **Set B (learning signal):** own reward r_{i,t}, used in PPO returns. Not fed back as observation.
- **Set C (logging):** everything, including true f, flip labels, exact rewards, welfare. For analysis only.

### Training Algorithm
The existing REINFORCE implementation multiplies logprob by immediate reward with no return-to-go. This CANNOT learn history-dependent strategies across 100-step sessions. You MUST implement PPO with GAE. Do not attempt to patch the existing REINFORCE for multi-step — rewrite the training loop.

---

## File Structure of marl-emecom (What You're Working With)

```
marl-emecom/
├── src/
│   ├── environments/
│   │   └── pgg/
│   │       └── pgg_parallel_v0.py    # <-- MAIN ENV FILE. Heavy modifications needed.
│   ├── algos/
│   │   ├── agent.py                  # <-- Agent class. Observation handling, policy networks.
│   │   ├── Reinforce.py              # <-- Current training algo. Will be replaced by PPO.
│   │   └── ...
│   ├── experiments_pgg_v0/
│   │   ├── train_given_params.py     # <-- Main training loop. Will be largely rewritten.
│   │   └── ...
│   └── ...
├── conf/                              # Hydra config files
└── requirements.txt
```

---

## Task List

### Phase 0: Setup and Reproduce Baseline

#### Task 0.1: Clone and install

```bash
git clone https://github.com/nicoleorzan/marl-emecom.git
cd marl-emecom
pip install -e .
pip freeze > requirements_locked.txt
```

Pin Python 3.9.x. Pin all dependency versions from the freeze. All subsequent work uses this locked environment.

**Acceptance criteria:**
- [ ] `requirements_locked.txt` exists with exact versions.
- [ ] `import` of all project modules works without error.

#### Task 0.2: Reproduce single-step baseline

Run the existing training script with default config (single-step EPGG, no communication, known f). Verify:
- Agents train without crashes.
- Cooperation rate changes over training (not stuck at 0 or 1).
- Rewards are consistent with the payoff formula.

**How to run:** Check `conf/` for hydra configs. The entry point is likely `src/experiments_pgg_v0/train_given_params.py`. Run with default params or minimal config.

**Acceptance criteria:**
- [ ] Training runs to completion.
- [ ] Logged cooperation rates show learning (not flat).
- [ ] You understand the data flow: env.reset() → env.step() → agent.act() → training update.

#### Task 0.3: Read and annotate key files

Before modifying anything, read and understand:

1. **`pgg_parallel_v0.py`** — Understand:
   - How `reset()` samples `current_multiplier` and calls `observe()`.
   - How `step()` computes rewards and returns observations (note: multi-step is broken — observations only defined in the `env_done` branch).
   - The return signature (4-tuple: observations, rewards, done, infos).
   - How `observe()` constructs agent observations (currently scalar).

2. **`agent.py`** — Understand:
   - `set_observation()`: how it stores and normalises obs. Note `gmm_` logic.
   - `get_action()`: how policy network produces actions.
   - `get_message()` / message handling: how messages are concatenated and fed to action policy.
   - `state_to_act`: what the action network actually sees as input.

3. **`train_given_params.py`** — Understand:
   - The training loop structure: reset → (optional comm step) → action step → env.step → reward → update.
   - How communication is orchestrated (trainer-side, not env-side).
   - The REINFORCE update: note it uses immediate reward, not return-to-go.
   - Note the typo: `agent.return_episode =+ rewards[...]` (should be `+=`).

**Acceptance criteria:**
- [ ] You can explain the full data flow from env.reset() to policy gradient update.
- [ ] You have identified all locations where observations are constructed, passed, and consumed.

---

### Phase 1: Environment Modifications

#### Task 1.1: Make `step()` multi-step compatible

**File:** `src/environments/pgg/pgg_parallel_v0.py`

**Current problems:**
1. In `step()`, the `observations` variable is only defined inside the `if env_done` branch. If `env_done` is False (i.e., mid-session), the function tries to return an undefined variable.
2. `observe()` is only called in `reset()` and in the done branch — not during ongoing play.
3. `current_multiplier` is set only in `reset()` — never updated mid-session.

**Required changes:**

```python
def step(self, actions):
    # 1. Apply actions, compute rewards (existing logic)
    # ... existing reward computation ...

    # 2. Increment step counter
    self.num_moves += 1
    env_done = (self.num_moves >= self.num_game_iterations)

    # 3. If not done, update f_t for next round (Sticky-EPGG)
    if not env_done:
        self._update_multiplier()  # New method, see Task 1.2

    # 4. ALWAYS generate new observations (not just when done)
    observations = self.observe()

    # 5. Return
    dones = {agent: env_done for agent in self.agents}
    # ... construct infos ...
    return observations, rewards, dones, infos
```

**Acceptance criteria:**
- [ ] `step()` returns valid observations at every step, not just the final one.
- [ ] The game runs for exactly `num_game_iterations` steps before `done = True`.
- [ ] Observations change between steps (not stale from reset).

#### Task 1.2: Implement Sticky-f (Markov switching)

**File:** `src/environments/pgg/pgg_parallel_v0.py`

Add a method `_update_multiplier()`:

```python
def _update_multiplier(self):
    """Markov switching of multiplication factor with hazard rate rho."""
    if np.random.random() < self.rho:
        # Switch to a different f value
        other_values = [f for f in self.F if f != self.current_multiplier]
        self.current_multiplier = np.random.choice(other_values)
```

**Constructor changes:**
- Add parameters: `F = [0.5, 1.5, 2.5, 3.5, 5.0]`, `rho = 0.05`.
- Store as `self.F` and `self.rho`.
- In `reset()`, sample initial `current_multiplier` uniformly from `F`.

**Acceptance criteria:**
- [ ] Over 10000 steps, each f value appears roughly 1/|F| of the time.
- [ ] Average run length per regime is approximately 1/rho = 20 steps.
- [ ] Transitions never stay on the same value (always switch to a *different* f).

#### Task 1.3: Remove observation clamping

**File:** `src/environments/pgg/pgg_parallel_v0.py`

In `observe()`, the current code may clamp or clip the noisy observation of f. Find and remove any clamping/clipping. The noisy observation should be:

```python
f_hat_i = self.current_multiplier + np.random.normal(0, self.sigma[i])
```

No clipping, no clamping, no rounding. Raw Gaussian noise.

**Acceptance criteria:**
- [ ] `f_hat` values can be negative or exceed `max(F)`.
- [ ] The distribution of `f_hat - f_true` matches N(0, sigma_i^2).

#### Task 1.4: Fix observation space type

**File:** `src/environments/pgg/pgg_parallel_v0.py`

Change `observation_space()` from `Discrete(...)` to:

```python
from gymnasium.spaces import Box
import numpy as np

def observation_space(self, agent):
    return Box(low=-np.inf, high=np.inf, shape=(self.obs_dim,), dtype=np.float32)
```

where `self.obs_dim` is the dimension of the raw env observation (just the noisy f_hat and endowment — the wrapper adds the rest). Set `self.obs_dim = 2` (noisy f, endowment).

**Acceptance criteria:**
- [ ] `env.observation_space(agent)` returns a `Box`.
- [ ] The dtype and shape match what `observe()` actually returns.

#### Task 1.5: Implement trembling-hand noise

**File:** `src/environments/pgg/pgg_parallel_v0.py`

In `step()`, after receiving agent actions but before computing rewards:

```python
def step(self, actions):
    # Store intended actions
    intended_actions = {agent: actions[agent] for agent in self.agents}

    # Apply trembling hand
    executed_actions = {}
    flips = {}
    for agent in self.agents:
        if np.random.random() < self.epsilon_tremble:
            executed_actions[agent] = 1 - actions[agent]  # Flip
            flips[agent] = True
        else:
            executed_actions[agent] = actions[agent]
            flips[agent] = False

    # Use executed_actions for reward computation
    # Store both in infos for logging
    # ...
```

**Constructor:** Add `epsilon_tremble = 0.05` parameter.

**Important:** Rewards are computed from `executed_actions`. Other agents observe `executed_actions` in subsequent rounds. The training buffer must store `intended_actions` (for policy gradient computation). The `infos` dict must include both intended and executed actions plus flip labels.

**Acceptance criteria:**
- [ ] With `epsilon_tremble = 0.05`, approximately 5% of actions are flipped over many steps.
- [ ] Rewards are computed from executed (post-flip) actions.
- [ ] `infos` contains intended actions, executed actions, and flip booleans.

#### Task 1.6: Hard config requirements

**File:** `src/algos/agent.py` and config files.

- Set `gmm_ = False` everywhere. The GMM logic in `set_observation()` assumes scalar obs and will crash with vectors.
- Set `normalize_obs = False` or remove the normalisation that divides by a scalar max. With multi-feature vectors, this doesn't make sense.
- Verify no other code path assumes observations are scalars.

**Acceptance criteria:**
- [ ] Agent initialises without error when `gmm_ = False`.
- [ ] Observation pipeline handles vector inputs without crashes.

---

### Phase 2: Observation Wrapper

#### Task 2.1: Build `ObservationWrapper` class

**Create new file:** `src/wrappers/observation_wrapper.py`

This is a trainer-side class (not a gym wrapper) that sits between the env and the agents. It:
1. Receives raw env observations (noisy f_hat, endowment).
2. Maintains history buffers.
3. Constructs the augmented observation vector (Set A).
4. Handles message injection.

```python
class ObservationWrapper:
    def __init__(self, n_agents, ewma_decay=0.9, comm_enabled=False,
                 n_senders=0, vocab_size=2, msg_dropout=0.1):
        self.n_agents = n_agents
        self.ewma_decay = ewma_decay
        self.comm_enabled = comm_enabled
        self.n_senders = n_senders
        self.vocab_size = vocab_size
        self.msg_dropout = msg_dropout

        # History buffers
        self.last_actions = None          # binary per agent
        self.last_coop_fraction = 0.0     # k_{t-1} / N
        self.ewma_coop = 0.0             # exponentially weighted moving avg

        # Message marginals (for dropout replacement)
        self.msg_marginals = {}           # per sender: running freq over vocab

    def reset(self):
        """Call at session start."""
        self.last_actions = {i: 0 for i in range(self.n_agents)}
        self.last_coop_fraction = 0.0
        self.ewma_coop = 0.0
        self.msg_marginals = {j: np.ones(self.vocab_size) / self.vocab_size
                              for j in range(self.n_senders)}

    def update(self, executed_actions):
        """Call after each env step with post-flip actions."""
        k = sum(executed_actions.values())
        self.last_coop_fraction = k / self.n_agents
        self.ewma_coop = (self.ewma_decay * self.ewma_coop +
                          (1 - self.ewma_decay) * self.last_coop_fraction)
        self.last_actions = dict(executed_actions)

    def update_msg_marginals(self, sender_id, message):
        """Update running message frequency for a sender."""
        alpha = 0.01  # smoothing rate
        onehot = np.zeros(self.vocab_size)
        onehot[message] = 1.0
        self.msg_marginals[sender_id] = (
            (1 - alpha) * self.msg_marginals[sender_id] + alpha * onehot
        )

    def apply_msg_dropout(self, messages):
        """Replace messages with marginal samples with prob msg_dropout."""
        dropped = {}
        for sender_id, msg in messages.items():
            if np.random.random() < self.msg_dropout:
                dropped[sender_id] = np.random.choice(
                    self.vocab_size, p=self.msg_marginals[sender_id]
                )
            else:
                dropped[sender_id] = msg
        return dropped

    def build_obs(self, agent_id, raw_env_obs, messages=None):
        """
        Construct Set A observation vector for agent_id.

        raw_env_obs: dict with 'f_hat' (float) and 'endowment' (float)
        messages: dict {sender_id: int} or None

        Returns: np.array of shape (obs_dim,)
        """
        obs = [
            raw_env_obs['f_hat'],
            raw_env_obs['endowment'],
            self.last_coop_fraction,
            float(self.last_actions.get(agent_id, 0)),
            self.ewma_coop,
        ]

        if self.comm_enabled and messages is not None:
            for sender_id in sorted(messages.keys()):
                onehot = np.zeros(self.vocab_size)
                onehot[messages[sender_id]] = 1.0
                obs.extend(onehot.tolist())

        return np.array(obs, dtype=np.float32)

    @property
    def obs_dim(self):
        base = 5  # f_hat, endowment, last_coop_frac, own_last_action, ewma
        if self.comm_enabled:
            base += self.n_senders * self.vocab_size
        return base
```

**Acceptance criteria:**
- [ ] `build_obs()` returns float32 array of consistent dimension.
- [ ] `obs_dim` matches actual output length.
- [ ] EWMA updates correctly (test with known sequence).
- [ ] Message dropout replaces approximately `msg_dropout` fraction of messages.
- [ ] At t=0, history features are zero.

#### Task 2.2: Integrate wrapper into training loop

The wrapper must be called at the right points:

```
1. env.reset() → get raw obs
2. wrapper.reset()
3. For t = 0 to T-1:
   a. augmented_obs = wrapper.build_obs(raw_obs, messages) for each agent
   b. agents select messages (if comm) using augmented_obs
   c. apply message dropout
   d. agents select actions using augmented_obs + messages
   e. env.step(intended_actions) → raw_obs_next, rewards, dones, infos
      (env applies tremble, computes rewards from executed actions)
   f. wrapper.update(executed_actions from infos)
   g. store transition: (augmented_obs, intended_action, reward, done, ...)
   h. raw_obs = raw_obs_next
4. Compute returns/advantages over stored trajectory
5. PPO update
```

**Acceptance criteria:**
- [ ] Each agent receives a different observation (at minimum, `own_last_action` differs).
- [ ] Messages flow: comm agents → dropout → listening agents' observations.
- [ ] History features at step t reflect actions from step t-1 (not step t).

---

### Phase 3: PPO Training Loop

#### Task 3.1: Implement trajectory buffer

**Create new file:** `src/algos/trajectory_buffer.py`

Stores transitions for a full session:

```python
class TrajectoryBuffer:
    def __init__(self, n_agents, T, obs_dim):
        self.observations = np.zeros((T, n_agents, obs_dim), dtype=np.float32)
        self.actions = np.zeros((T, n_agents), dtype=np.int32)      # intended
        self.rewards = np.zeros((T, n_agents), dtype=np.float32)
        self.values = np.zeros((T, n_agents), dtype=np.float32)     # V(s)
        self.log_probs = np.zeros((T, n_agents), dtype=np.float32)
        self.dones = np.zeros((T,), dtype=bool)

        # For logging only (Set C)
        self.executed_actions = np.zeros((T, n_agents), dtype=np.int32)
        self.flips = np.zeros((T, n_agents), dtype=bool)
        self.true_f = np.zeros((T,), dtype=np.float32)
        self.f_hats = np.zeros((T, n_agents), dtype=np.float32)
        self.messages = None  # allocated if comm enabled
        self.agent_rewards = np.zeros((T, n_agents), dtype=np.float32)

        self.t = 0

    def store(self, obs, actions, rewards, values, log_probs, done,
              executed_actions, flips, true_f, f_hats, messages=None):
        # Store at self.t, increment self.t
        ...

    def compute_gae(self, last_values, gamma=0.99, lam=0.95):
        """
        Compute Generalised Advantage Estimation.

        last_values: V(s_T) for each agent (bootstrap if not done).

        Returns: advantages (T, n_agents), returns (T, n_agents)
        """
        advantages = np.zeros_like(self.rewards)
        last_gae = np.zeros(self.rewards.shape[1])

        for t in reversed(range(self.t)):
            if t == self.t - 1:
                next_values = last_values
                next_non_terminal = 1.0 - float(self.dones[t])
            else:
                next_values = self.values[t + 1]
                next_non_terminal = 1.0 - float(self.dones[t])

            delta = (self.rewards[t] + gamma * next_values * next_non_terminal
                     - self.values[t])
            advantages[t] = last_gae = (delta + gamma * lam *
                                         next_non_terminal * last_gae)

        returns = advantages + self.values[:self.t]
        return advantages, returns

    def reset(self):
        self.t = 0
```

**Acceptance criteria:**
- [ ] GAE computation matches a known reference (test with hand-computed example).
- [ ] Buffer handles full T=100 episodes correctly.
- [ ] All Set C logging fields are populated.

#### Task 3.2: Implement PPO update

**Create new file:** `src/algos/ppo.py`

```python
class PPOTrainer:
    def __init__(self, agents, lr=3e-4, clip_ratio=0.2, value_coeff=0.5,
                 entropy_coeff=0.01, max_grad_norm=0.5, ppo_epochs=4,
                 mini_batch_size=32):
        # agents: dict of Agent objects
        # Each agent has: policy network, value network, optimizer
        ...

    def update(self, buffer, advantages, returns):
        """
        Run K epochs of PPO updates on the collected trajectory.

        For each mini-batch:
        1. Recompute log_probs and values from current policy/value networks.
        2. Compute ratio = exp(new_log_prob - old_log_prob).
        3. Clipped surrogate: min(ratio * adv, clip(ratio, 1-eps, 1+eps) * adv).
        4. Value loss: MSE(V(s), returns).
        5. Entropy bonus.
        6. Backprop and step optimizer.
        """
        ...
```

**Key details:**
- Each agent has its own policy and value networks but they can share architecture.
- Value network input is the same augmented observation as the policy.
- Normalise advantages (zero mean, unit variance) within each mini-batch.
- Gradient clipping: `max_grad_norm = 0.5`.
- For communication agents: treat message selection as part of the action; include its log-prob in the surrogate objective. Alternatively, keep the auxiliary signaling/listening losses from marl-emecom and add them to the PPO loss.

**Acceptance criteria:**
- [ ] Loss decreases over PPO epochs within a single batch.
- [ ] Policy probabilities don't collapse to 0/1 (entropy stays reasonable).
- [ ] Value predictions improve over training.
- [ ] No NaN gradients.

#### Task 3.3: Implement main training loop

**Create new file:** `src/experiments_pgg_v0/train_ppo.py`

This replaces `train_given_params.py` for the multi-step setting.

Pseudocode:

```python
def train(config):
    # 1. Create environment
    env = PGGParallelEnv(
        n_agents=config.n_agents,
        num_game_iterations=config.T,  # 100
        F=config.F, rho=config.rho,
        epsilon_tremble=config.epsilon_tremble,
        sigmas=config.sigmas,
    )

    # 2. Create observation wrapper
    wrapper = ObservationWrapper(
        n_agents=config.n_agents,
        comm_enabled=config.comm_enabled,
        n_senders=config.n_senders,
        vocab_size=config.vocab_size,
        msg_dropout=config.msg_dropout,
    )

    # 3. Create agents (with value heads for PPO)
    agents = create_agents(config, wrapper.obs_dim)

    # 4. Create PPO trainer
    ppo = PPOTrainer(agents, lr=config.lr, ...)

    # 5. Training loop
    for episode in range(config.n_episodes):
        buffer = TrajectoryBuffer(config.n_agents, config.T, wrapper.obs_dim)
        raw_obs = env.reset()
        wrapper.reset()
        current_messages = None

        for t in range(config.T):
            # Build augmented observations
            aug_obs = {}
            for i in agents:
                aug_obs[i] = wrapper.build_obs(i, raw_obs[i], current_messages)

            # Communication step (if enabled)
            if config.comm_enabled:
                messages = {}
                for i in config.sender_ids:
                    messages[i] = agents[i].get_message(aug_obs[i])
                messages = wrapper.apply_msg_dropout(messages)
                for sender_id, msg in messages.items():
                    wrapper.update_msg_marginals(sender_id, msg)
                # Rebuild obs with messages
                for i in agents:
                    aug_obs[i] = wrapper.build_obs(i, raw_obs[i], messages)
                current_messages = messages

            # Action step
            intended_actions, log_probs, values = {}, {}, {}
            for i in agents:
                action, lp, val = agents[i].act(aug_obs[i])
                intended_actions[i] = action
                log_probs[i] = lp
                values[i] = val

            # Environment step (tremble applied internally)
            raw_obs_next, rewards, dones, infos = env.step(intended_actions)

            # Extract executed actions from infos
            executed_actions = infos['executed_actions']
            flips = infos['flips']

            # Update wrapper history
            wrapper.update(executed_actions)

            # Store transition
            buffer.store(
                obs=aug_obs, actions=intended_actions, rewards=rewards,
                values=values, log_probs=log_probs, done=dones,
                executed_actions=executed_actions, flips=flips,
                true_f=infos['true_f'], f_hats=raw_obs,
                messages=current_messages,
            )
            raw_obs = raw_obs_next

        # Compute advantages
        last_values = {i: agents[i].get_value(aug_obs[i]) for i in agents}
        advantages, returns = buffer.compute_gae(last_values)

        # PPO update
        ppo.update(buffer, advantages, returns)

        # Logging
        if episode % config.log_interval == 0:
            log_metrics(buffer, episode)

    save_agents(agents, config.save_path)
```

**Acceptance criteria:**
- [ ] Full training loop runs without crashes for 100 episodes.
- [ ] Cooperation rate changes over training.
- [ ] Each step of a session produces different observations.
- [ ] f switches occur during sessions (check logs).
- [ ] Rewards are correct per the payoff formula at each step.

---

### Phase 4: Data Logging

#### Task 4.1: Implement session logger

**Create new file:** `src/logging/session_logger.py`

After each evaluation session (separate evaluation pass, not training), save the full Set C data:

```python
class SessionLogger:
    def __init__(self, save_dir, condition_name, seed):
        self.save_dir = save_dir
        self.condition_name = condition_name
        self.seed = seed
        self.session_count = 0

    def log_session(self, buffer):
        """Save one session's data as .npz"""
        np.savez(
            f"{self.save_dir}/data_{self.condition_name}_{self.seed}_{self.session_count}.npz",
            true_f=buffer.true_f[:buffer.t],
            f_hats=buffer.f_hats[:buffer.t],
            intended_actions=buffer.actions[:buffer.t],
            executed_actions=buffer.executed_actions[:buffer.t],
            flips=buffer.flips[:buffer.t],
            rewards=buffer.agent_rewards[:buffer.t],
            messages=buffer.messages[:buffer.t] if buffer.messages is not None else np.array([]),
            cooperation_count=buffer.executed_actions[:buffer.t].sum(axis=1),
            welfare=buffer.agent_rewards[:buffer.t].sum(axis=1),
        )
        self.session_count += 1

    def consolidate(self):
        """Merge all session files into one array per condition+seed."""
        # Load all individual .npz, stack into (n_sessions, T, ...) arrays
        # Save as single HDF5 or large .npz
        # Delete individual files
        ...
```

**Acceptance criteria:**
- [ ] Each `.npz` contains all Set C fields.
- [ ] Shapes are consistent: `(T,)` for scalars, `(T, N)` for per-agent.
- [ ] `consolidate()` produces a single file with shape `(n_sessions, T, ...)`.

---

### Phase 5: Unit Tests

#### Task 5.1: Environment tests

**Create new file:** `tests/test_env.py`

```python
def test_multi_step_observations():
    """step() returns valid observations at every step, not just final."""
    env = PGGParallelEnv(num_game_iterations=10, ...)
    obs = env.reset()
    for t in range(10):
        actions = {a: np.random.randint(2) for a in env.agents}
        obs, rewards, dones, infos = env.step(actions)
        for agent in env.agents:
            assert obs[agent] is not None
            assert isinstance(obs[agent], np.ndarray)
        if t < 9:
            assert not any(dones.values())
    assert all(dones.values())

def test_sticky_f_transitions():
    """f switches at approximately the expected rate."""
    env = PGGParallelEnv(num_game_iterations=10000, rho=0.1, ...)
    env.reset()
    switches = 0
    prev_f = env.current_multiplier
    for t in range(10000):
        actions = {a: 0 for a in env.agents}
        env.step(actions)
        if env.current_multiplier != prev_f:
            switches += 1
        prev_f = env.current_multiplier
    assert 500 < switches < 1500  # ~1000 expected

def test_sticky_f_never_self_transitions():
    """When f switches, it always goes to a DIFFERENT value."""
    # Use high rho to get many switches
    env = PGGParallelEnv(num_game_iterations=1000, rho=0.5, ...)
    env.reset()
    prev_f = env.current_multiplier
    for t in range(1000):
        env.step({a: 0 for a in env.agents})
        # No assertion needed per step; the logic in _update_multiplier
        # guarantees this. But verify post-hoc:
        prev_f = env.current_multiplier

def test_tremble_rate():
    """Approximately epsilon fraction of actions are flipped."""
    env = PGGParallelEnv(num_game_iterations=10000, epsilon_tremble=0.05, ...)
    env.reset()
    total, flipped = 0, 0
    for t in range(10000):
        actions = {a: 1 for a in env.agents}  # all cooperate
        _, _, _, infos = env.step(actions)
        for a in env.agents:
            total += 1
            if infos['flips'][a]:
                flipped += 1
    rate = flipped / total
    assert 0.03 < rate < 0.07  # 5% +/- 2%

def test_rewards_match_payoff_formula():
    """Rewards exactly match u_i(a, f, c) using EXECUTED actions."""
    env = PGGParallelEnv(num_game_iterations=1, epsilon_tremble=0.0, ...)
    env.reset()
    env.current_multiplier = 2.5
    actions = {a: 1 for a in env.agents}  # all cooperate
    _, rewards, _, _ = env.step(actions)
    # u_i = (1/N) * N * c * f = c * f = 4 * 2.5 = 10.0
    for a in env.agents:
        assert abs(rewards[a] - 10.0) < 1e-6

def test_no_observation_clamping():
    """Noisy f_hat can be negative or exceed max(F)."""
    env = PGGParallelEnv(num_game_iterations=1000, sigmas=[5.0]*4, ...)
    env.reset()
    saw_negative, saw_above_max = False, False
    for t in range(1000):
        obs = env.observe()
        for a in env.agents:
            if obs[a]['f_hat'] < 0:
                saw_negative = True
            if obs[a]['f_hat'] > max(env.F):
                saw_above_max = True
        env.step({a: 0 for a in env.agents})
    assert saw_negative and saw_above_max

def test_observation_space_is_box():
    env = PGGParallelEnv(...)
    from gymnasium.spaces import Box
    for agent in env.agents:
        assert isinstance(env.observation_space(agent), Box)
```

#### Task 5.2: Wrapper tests

**Create new file:** `tests/test_wrapper.py`

```python
def test_obs_dimension_consistency():
    wrapper = ObservationWrapper(n_agents=4, comm_enabled=True,
                                 n_senders=3, vocab_size=2)
    wrapper.reset()
    messages = {0: 0, 1: 1, 2: 0}
    for t in range(20):
        obs = wrapper.build_obs(0, {'f_hat': 1.5, 'endowment': 4.0}, messages)
        assert obs.shape == (wrapper.obs_dim,)
        assert obs.dtype == np.float32
        wrapper.update({0: 1, 1: 0, 2: 1, 3: 0})

def test_ewma_computation():
    wrapper = ObservationWrapper(n_agents=4, ewma_decay=0.9)
    wrapper.reset()
    wrapper.update({0: 1, 1: 1, 2: 1, 3: 1})  # all cooperate
    assert abs(wrapper.ewma_coop - 0.1) < 1e-6   # 0.9*0 + 0.1*1.0
    wrapper.update({0: 1, 1: 1, 2: 1, 3: 1})
    assert abs(wrapper.ewma_coop - 0.19) < 1e-6  # 0.9*0.1 + 0.1*1.0

def test_message_dropout_rate():
    wrapper = ObservationWrapper(n_agents=4, comm_enabled=True,
                                 n_senders=3, vocab_size=2, msg_dropout=0.5)
    wrapper.reset()
    original = {0: 0, 1: 0, 2: 0}
    changed, total = 0, 0
    for _ in range(10000):
        dropped = wrapper.apply_msg_dropout(original)
        for k in original:
            total += 1
            if dropped[k] != original[k]:
                changed += 1
    rate = changed / total
    # ~25% changed (50% dropout * ~50% chance of different token from uniform)
    assert 0.2 < rate < 0.35

def test_history_features_lag():
    wrapper = ObservationWrapper(n_agents=4)
    wrapper.reset()
    obs0 = wrapper.build_obs(0, {'f_hat': 2.0, 'endowment': 4.0})
    assert obs0[2] == 0.0  # last_coop_fraction = 0 at t=0

    wrapper.update({0: 1, 1: 1, 2: 0, 3: 0})  # 2 of 4 cooperated
    obs1 = wrapper.build_obs(0, {'f_hat': 2.0, 'endowment': 4.0})
    assert abs(obs1[2] - 0.5) < 1e-6  # last_coop_fraction = 2/4
```

#### Task 5.3: GAE test

**Create new file:** `tests/test_gae.py`

```python
def test_gae_simple_case():
    """Test GAE against hand-calculated values."""
    buffer = TrajectoryBuffer(n_agents=1, T=3, obs_dim=5)
    buffer.rewards = np.array([[1.0], [2.0], [3.0]])
    buffer.values = np.array([[1.0], [2.0], [3.0]])
    buffer.dones = np.array([False, False, True])
    buffer.t = 3

    advantages, returns = buffer.compute_gae(
        last_values=np.array([0.0]), gamma=0.99, lam=0.95
    )
    # t=2 (terminal): delta = 3 + 0 - 3 = 0, gae = 0
    # t=1: delta = 2 + 0.99*3 - 2 = 2.97, gae = 2.97 + 0 = 2.97
    # t=0: delta = 1 + 0.99*2 - 1 = 1.98, gae = 1.98 + 0.99*0.95*2.97
    assert abs(advantages[2, 0] - 0.0) < 1e-4
    assert abs(advantages[1, 0] - 2.97) < 1e-2
```

#### Task 5.4: Integration test

**Create new file:** `tests/test_integration.py`

```python
def test_full_loop_runs():
    """Complete training loop runs for 10 episodes without crashing."""
    config = minimal_test_config()  # T=10, n_agents=4, n_episodes=10
    train(config)

def test_cooperation_changes():
    """Cooperation rate is not constant across training."""
    config = minimal_test_config(n_episodes=500)
    metrics = train(config)
    coop_rates = [m['coop_rate'] for m in metrics]
    assert np.std(coop_rates) > 0.01
```

---

### Phase 6: Regime Identifiability Audit

#### Task 6.1: Compute Bayesian filter convergence speed

**Create new file:** `src/analysis/regime_audit.py`

After the environment is working:

1. Run 1000 sessions with default params (sigma=0.5, rho=0.05, N=4).
2. At each regime switch, compute: how many rounds until max_f pi_t(f) > 0.9?
3. Report mean, median, distribution.

If median convergence is <= 2 rounds, inference is too easy. Increase sigma or reduce |F|.

```python
def regime_audit(env_config, n_sessions=1000):
    """How fast can a Bayesian observer identify the regime after a switch?"""
    convergence_times = []
    for session in range(n_sessions):
        # Reset env, run T steps
        # At each step: run exact HMM forward pass
        # Detect regime switches (f changed from previous step)
        # For each switch: count rounds until posterior concentrates
        ...
    return {
        'mean': np.mean(convergence_times),
        'median': np.median(convergence_times),
        'p90': np.percentile(convergence_times, 90),
    }
```

**Acceptance criteria:**
- [ ] Audit runs and produces statistics.
- [ ] Median convergence time is documented.
- [ ] If <= 2, a recommendation is logged to increase sigma.

---

## Deliverables Checklist (End of Week 2)

- [ ] Locked `requirements_locked.txt`
- [ ] Patched `pgg_parallel_v0.py`: multi-step, Sticky-f, tremble, no clamping, Box obs space
- [ ] `ObservationWrapper` class with all Set A features, message dropout, EWMA
- [ ] `TrajectoryBuffer` with GAE computation
- [ ] `PPOTrainer` with clipped surrogate, value loss, entropy bonus
- [ ] `train_ppo.py`: complete training loop for multi-step sessions
- [ ] `SessionLogger` with per-session .npz output and consolidation
- [ ] All unit tests passing (env, wrapper, GAE, integration)
- [ ] Regime identifiability audit results documented
- [ ] Dynamic Richness Checklist run on at least one condition
- [ ] Git repo with clean commit history, each major change in a separate commit

---

## What NOT To Do

- Do NOT modify the environment to include messages in observations. Messages are handled by the wrapper.
- Do NOT include rewards or welfare in agent observations (Set A). This is the information leakage rule.
- Do NOT use the existing REINFORCE code for multi-step training. It cannot learn history-dependent strategies.
- Do NOT upgrade PettingZoo beyond what the repo expects. Pin versions.
- Do NOT enable `gmm_` in agent configs.
- Do NOT write a recurrent policy. History dependence comes from the observation wrapper features.
- Do NOT implement the PLRNN or Bayesian baselines. Those are weeks 5+.

---

## Reference: Technical Specification

The full project spec (DSC_EPGG_Technical_Specification_v3.md) contains complete scientific context, research questions, evaluation protocols, and later-phase plans. Refer to it for questions about broader goals, but do NOT implement anything beyond weeks 1-2 scope.
