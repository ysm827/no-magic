"""
The ReAct reasoning loop from first principles: Thought -> Action -> Observation as a
trainable state machine -- showing how structured reasoning emerges from interleaving
thinking with environment interaction.
"""
# Reference: Yao et al., "ReAct: Synergizing Reasoning and Acting in Language Models" (2022)
# https://arxiv.org/abs/2210.03629

# === TRADEOFFS ===
# + Interleaving thought and action enables multi-step reasoning
# + Observations ground the agent's reasoning in real state (reduces hallucination)
# + Fully observable trace: every decision is explainable
# - Requires well-defined tool interfaces (brittle if tools fail)
# - Sequential execution: no parallelism in the reasoning chain
# - Performance depends heavily on the quality of the reasoning policy
# WHEN TO USE: Multi-step tasks requiring tool use, question answering
#   with retrieval, any task where reasoning traces improve reliability.
# WHEN NOT TO: Simple single-step tasks, real-time applications where
#   the thought-action loop adds unacceptable latency.

from __future__ import annotations

import math
import random
import time

random.seed(42)


# === CONSTANTS AND HYPERPARAMETERS ===

# Policy network: maps state features to a distribution over actions.
# Two-layer MLP trained with REINFORCE policy gradients.
STATE_DIM = 16        # compact state vector
HIDDEN_DIM = 32       # single hidden layer
ACTION_DIM = 7        # see ACTION SPACE section

# Training
LEARNING_RATE = 0.10  # high LR because action space is tiny and reward is dense
NUM_EPOCHS = 500      # more epochs compensate for REINFORCE variance
BASELINE_DECAY = 0.95 # EMA baseline for variance reduction
ENTROPY_COEFF = 0.005 # very small entropy: with masking, we want fast convergence

# Agent
MAX_STEPS = 3         # exactly 3 steps needed: lookup, lookup, compute

# Signpost: Production ReAct agents (ChatGPT plugins, LangChain) use LLMs with
# billions of parameters as the policy. Our ~1800-parameter MLP learns the same
# Thought -> Action -> Observation structure. The key insight is identical:
# structured interleaving of reasoning and grounded observation beats pure
# reasoning or pure action.


# === KNOWLEDGE BASE AND TOOLS ===

# A small fact store the agent queries to answer multi-hop questions.
# The agent cannot see all facts at once -- it must choose which entity to look up.
# This mirrors how real ReAct agents interact with APIs: each tool call reveals
# partial information, forcing deliberate sequential reasoning.

KNOWLEDGE_BASE: dict[str, dict[str, str | int | float]] = {
    "earth": {
        "type": "planet", "radius_km": 6371, "mass_kg": 5.97e24,
        "moons": 1, "orbital_period_days": 365, "distance_from_sun_au": 1.0,
    },
    "mars": {
        "type": "planet", "radius_km": 3390, "mass_kg": 6.42e23,
        "moons": 2, "orbital_period_days": 687, "distance_from_sun_au": 1.52,
    },
    "jupiter": {
        "type": "planet", "radius_km": 69911, "mass_kg": 1.90e27,
        "moons": 95, "orbital_period_days": 4333, "distance_from_sun_au": 5.20,
    },
    "saturn": {
        "type": "planet", "radius_km": 58232, "mass_kg": 5.68e26,
        "moons": 146, "orbital_period_days": 10759, "distance_from_sun_au": 9.54,
    },
    "mercury": {
        "type": "planet", "radius_km": 2440, "mass_kg": 3.30e23,
        "moons": 0, "orbital_period_days": 88, "distance_from_sun_au": 0.39,
    },
    "venus": {
        "type": "planet", "radius_km": 6052, "mass_kg": 4.87e24,
        "moons": 0, "orbital_period_days": 225, "distance_from_sun_au": 0.72,
    },
    "neptune": {
        "type": "planet", "radius_km": 24622, "mass_kg": 1.02e26,
        "moons": 16, "orbital_period_days": 60190, "distance_from_sun_au": 30.07,
    },
    "uranus": {
        "type": "planet", "radius_km": 25362, "mass_kg": 8.68e25,
        "moons": 28, "orbital_period_days": 30687, "distance_from_sun_au": 19.19,
    },
    "sun": {
        "type": "star", "radius_km": 696340, "mass_kg": 1.99e30,
        "surface_temp_k": 5778,
    },
    "moon": {
        "type": "satellite", "radius_km": 1737, "mass_kg": 7.34e22,
        "orbital_period_days": 27, "parent": "earth",
    },
}

ENTITY_LIST = sorted(KNOWLEDGE_BASE.keys())


def tool_lookup(entity: str) -> dict[str, str | int | float] | str:
    """Retrieve all facts about an entity from the knowledge base.

    This is the agent's primary information-gathering tool. It cannot see the
    knowledge base directly -- each lookup is a deliberate choice that reveals
    partial information, forcing multi-step reasoning.
    """
    entity = entity.lower().strip()
    if entity in KNOWLEDGE_BASE:
        return KNOWLEDGE_BASE[entity]
    return f"Entity '{entity}' not found"


def tool_calculate(a: float, b: float, op: str) -> float | str:
    """Evaluate binary arithmetic. Production agents use sandboxed code execution;
    we restrict to basic ops for a clean, learnable interface."""
    if op == "+":
        return a + b
    elif op == "-":
        return abs(a - b)
    elif op == "/":
        return a / b if b != 0 else "Division by zero"
    return f"Unknown op: {op}"


def tool_compare(a: float, b: float) -> str:
    """Compare two numeric values. Returns which is greater/less/equal."""
    if a > b:
        return "first_greater"
    elif a < b:
        return "second_greater"
    return "equal"


# === QUESTION SET ===

# Each question requires exactly 2-3 tool calls to answer:
#   1. lookup(entity_1) -> get value_1
#   2. lookup(entity_2) -> get value_2
#   3. compare(v1, v2) OR calculate(v1, v2, op)
#
# Why multi-hop? Single-lookup questions don't need ReAct's structured reasoning.
# The 2-lookup + 1-compute pattern forces the agent to learn a specific tool
# calling sequence: gather information first, then compute.

Question = dict

QUESTIONS: list[Question] = [
    {
        "text": "Which planet has more moons, Mars or Jupiter?",
        "entities": ["mars", "jupiter"], "attribute": "moons",
        "answer_type": "comparison", "answer": "jupiter",
    },
    {
        "text": "What is the ratio of Jupiter's radius to Earth's radius?",
        "entities": ["jupiter", "earth"], "attribute": "radius_km",
        "answer_type": "ratio", "answer": 10.97,
    },
    {
        "text": "Which planet is closer to the Sun, Venus or Mercury?",
        "entities": ["venus", "mercury"], "attribute": "distance_from_sun_au",
        "answer_type": "comparison_min", "answer": "mercury",
    },
    {
        "text": "How many total moons do Earth and Mars have combined?",
        "entities": ["earth", "mars"], "attribute": "moons",
        "answer_type": "sum", "answer": 3,
    },
    {
        "text": "Which is larger, Saturn or Jupiter?",
        "entities": ["saturn", "jupiter"], "attribute": "radius_km",
        "answer_type": "comparison", "answer": "jupiter",
    },
    {
        "text": "What is the difference in orbital period between Mars and Earth?",
        "entities": ["mars", "earth"], "attribute": "orbital_period_days",
        "answer_type": "difference", "answer": 322,
    },
    {
        "text": "Which has more moons, Saturn or Uranus?",
        "entities": ["saturn", "uranus"], "attribute": "moons",
        "answer_type": "comparison", "answer": "saturn",
    },
    {
        "text": "What is the sum of Mercury's and Venus's orbital periods?",
        "entities": ["mercury", "venus"], "attribute": "orbital_period_days",
        "answer_type": "sum", "answer": 313,
    },
    {
        "text": "Which planet has a longer year, Neptune or Uranus?",
        "entities": ["neptune", "uranus"], "attribute": "orbital_period_days",
        "answer_type": "comparison", "answer": "neptune",
    },
    {
        "text": "How many times more massive is Jupiter than Mars?",
        "entities": ["jupiter", "mars"], "attribute": "mass_kg",
        "answer_type": "ratio", "answer": 2959.5,
    },
    {
        "text": "Which is further from the Sun, Saturn or Jupiter?",
        "entities": ["saturn", "jupiter"], "attribute": "distance_from_sun_au",
        "answer_type": "comparison", "answer": "saturn",
    },
    {
        "text": "What is the total number of moons for Jupiter and Saturn?",
        "entities": ["jupiter", "saturn"], "attribute": "moons",
        "answer_type": "sum", "answer": 241,
    },
    {
        "text": "Which planet's radius is larger, Earth or Venus?",
        "entities": ["earth", "venus"], "attribute": "radius_km",
        "answer_type": "comparison", "answer": "earth",
    },
    {
        "text": "What is the ratio of the Sun's radius to Jupiter's radius?",
        "entities": ["sun", "jupiter"], "attribute": "radius_km",
        "answer_type": "ratio", "answer": 9.96,
    },
    {
        "text": "Does Mars or Earth have a longer orbital period?",
        "entities": ["mars", "earth"], "attribute": "orbital_period_days",
        "answer_type": "comparison", "answer": "mars",
    },
    {
        "text": "What is the combined mass of Earth and Moon?",
        "entities": ["earth", "moon"], "attribute": "mass_kg",
        "answer_type": "sum", "answer": 6.04e24,
    },
]


# === ACTION SPACE ===

# The action space mirrors ReAct's decision hierarchy:
#
#   Actions 0-1: LOOKUP -- which entity to retrieve?
#     0: lookup(entity_1)    1: lookup(entity_2)
#
#   Actions 2-6: COMPUTE -- what operation to apply to retrieved values?
#     2: compare(v1, v2)     3: sum(v1, v2)    4: |v1 - v2|
#     5: v1 / v2             6: v2 / v1
#
# Action masking prevents invalid actions: compute actions are masked until
# both values are retrieved; lookup actions are masked for already-retrieved
# entities. This is standard practice in RL for structured action spaces
# (Huang & Ontanon, 2020) and dramatically improves learning efficiency.
# Without masking, the agent wastes most of training re-looking up entities
# or computing with missing values.

ACTION_NAMES = [
    "lookup(entity_1)", "lookup(entity_2)",
    "compare", "sum", "difference", "ratio_1/2", "ratio_2/1",
]


def get_action_mask(
    question: Question,
    retrieved_values: dict[str, float | int | None],
) -> list[bool]:
    """Compute which actions are valid given the current state.

    Mask rules:
      - lookup(entity_i): valid only if entity_i has NOT been looked up yet
      - compute ops: valid only if BOTH entities have been looked up
      - At least one action is always valid (prevents degenerate states)
    """
    entities = question["entities"]
    has_v1 = retrieved_values.get(entities[0]) is not None
    has_v2 = retrieved_values.get(entities[1]) is not None
    both = has_v1 and has_v2

    mask = [
        not has_v1,          # lookup entity_1: only if not yet retrieved
        not has_v2,          # lookup entity_2: only if not yet retrieved
        both,                # compare: need both values
        both,                # sum: need both values
        both,                # difference: need both values
        both,                # ratio_1/2: need both values
        both,                # ratio_2/1: need both values
    ]

    # Safety: ensure at least one action is valid
    if not any(mask):
        mask = [True] * ACTION_DIM

    return mask


def execute_action(
    action: int,
    question: Question,
    retrieved_values: dict[str, float | int | None],
) -> dict:
    """Execute a tool call and return the observation.

    Maps discrete action index to a concrete tool invocation. The tool's output
    becomes the "Observation" in the ReAct trace, grounding the next Thought step.
    """
    entities = question["entities"]
    attr = question["attribute"]

    if action == 0:  # lookup entity_1
        entity = entities[0]
        result = tool_lookup(entity)
        value = result[attr] if isinstance(result, dict) and attr in result else None
        return {"tool": "lookup", "entity": entity, "value": value, "result": result}

    elif action == 1:  # lookup entity_2
        entity = entities[1]
        result = tool_lookup(entity)
        value = result[attr] if isinstance(result, dict) and attr in result else None
        return {"tool": "lookup", "entity": entity, "value": value, "result": result}

    else:  # compute operations (actions 2-6)
        v1 = retrieved_values.get(entities[0])
        v2 = retrieved_values.get(entities[1])
        if v1 is None or v2 is None:
            return {"tool": "compute", "result": "missing_values", "value": None}

        fv1, fv2 = float(v1), float(v2)
        if action == 2:
            result = tool_compare(fv1, fv2)
            return {"tool": "compare", "a": v1, "b": v2, "result": result, "value": None}
        elif action == 3:
            result = tool_calculate(fv1, fv2, "+")
            return {"tool": "calculate", "op": "+", "a": fv1, "b": fv2, "result": result, "value": result}
        elif action == 4:
            result = tool_calculate(fv1, fv2, "-")
            return {"tool": "calculate", "op": "-", "a": fv1, "b": fv2, "result": result, "value": result}
        elif action == 5:
            result = tool_calculate(fv1, fv2, "/")
            return {"tool": "calculate", "op": "/", "a": fv1, "b": fv2, "result": result, "value": result}
        elif action == 6:
            result = tool_calculate(fv2, fv1, "/")
            return {"tool": "calculate", "op": "/", "a": fv2, "b": fv1, "result": result, "value": result}

    return {"tool": "noop", "result": "invalid", "value": None}


# === STATE ENCODING ===

# The state vector is the agent's "perception" -- it encodes everything the
# agent needs to decide what to do next. In text-based ReAct, the LLM reads
# the full conversation history. Here, we compress the relevant information
# into a fixed-size vector that the policy network processes.
#
# Critical design choice: the state must clearly distinguish "need to look up
# entity_1" from "need to look up entity_2" from "both looked up, need to compute".
# This three-phase structure (gather, gather, compute) is the backbone of
# multi-hop reasoning.

def encode_state(
    question: Question,
    step: int,
    retrieved_values: dict[str, float | int | None],
) -> list[float]:
    """Build the state vector for the policy network.

    Layout (STATE_DIM = 16):
      [0:5]    question type one-hot (comparison, comparison_min, sum, diff, ratio)
      [5]      entity_1 looked up? (0 or 1)
      [6]      entity_2 looked up? (0 or 1)
      [7:9]    retrieved values (log-normalized, 0 if not yet retrieved)
      [9]      step / MAX_STEPS (progress)
      [10:16]  entity pair encoding (which pair of entities is in the question)
    """
    state = [0.0] * STATE_DIM
    entities = question["entities"]

    # Question type one-hot: tells the policy which compute operation to use
    # This is the most important feature -- it directly determines the final action.
    type_map = {"comparison": 0, "comparison_min": 1, "sum": 2, "difference": 3, "ratio": 4}
    atype = question.get("answer_type", "comparison")
    state[type_map.get(atype, 0)] = 1.0

    # Lookup completion flags: the policy's primary signal for phase transitions.
    # Phase 1: both 0 -> do a lookup.  Phase 2: one is 1 -> do the other lookup.
    # Phase 3: both 1 -> compute.
    has_v1 = retrieved_values.get(entities[0]) is not None
    has_v2 = retrieved_values.get(entities[1]) is not None
    state[5] = 1.0 if has_v1 else 0.0
    state[6] = 1.0 if has_v2 else 0.0

    # Retrieved values: log-scale normalization handles the huge range
    # (moons=0 vs mass=1.9e30). The policy doesn't need exact values --
    # it just needs to know they exist and their rough magnitude.
    for i, ent in enumerate(entities[:2]):
        val = retrieved_values.get(ent)
        if val is not None and isinstance(val, (int, float)):
            state[7 + i] = math.log(abs(float(val)) + 1) / 70.0

    # Step counter: normalized progress through the episode
    state[9] = step / MAX_STEPS

    # Question type amplification when both values are ready.
    # When both entities have been looked up, the ONLY remaining decision is
    # which compute operation to use. We amplify the question type signal
    # to make this decision easier for the policy. This is analogous to how
    # attention mechanisms let transformers focus on relevant context.
    if has_v1 and has_v2:
        for i in range(5):
            state[i] *= 3.0  # triple the type signal in compute phase

    # Entity pair encoding: hashes the entity pair into feature slots.
    # Different entity pairs activate different features, allowing the policy
    # to learn entity-specific patterns.
    for i, ent in enumerate(entities[:2]):
        ent_idx = ENTITY_LIST.index(ent) if ent in ENTITY_LIST else 0
        slot = 10 + (ent_idx * 3 + i) % 6
        state[slot] = (ent_idx + 1) / len(ENTITY_LIST)

    return state


# === POLICY NETWORK ===

# Two-layer MLP with masked softmax output. The masking is key: it constrains
# the agent to only select valid actions at each step, eliminating the need
# to learn "don't compute before looking up" from scratch.
#
# Architecture: state -> ReLU hidden -> masked softmax -> action probs
#
# Trained with REINFORCE (Williams, 1992):
#   ∇J(θ) = E[ ∇log π(a|s) * (R - b) ]
#
# The gradient of log π for masked softmax is identical to standard softmax
# over the valid action subset. Masked-out actions get zero gradient.
#
# Signpost: This is a tiny version of what happens inside function-calling LLMs.
# The LLM's output distribution over tokens is constrained (masked) to only
# produce valid function signatures. Our discrete action masking is the same
# principle applied to a simpler action space.

def init_policy() -> dict[str, list]:
    """Initialize weights with Xavier initialization: std = sqrt(2/(fan_in+fan_out))."""
    std1 = math.sqrt(2.0 / (STATE_DIM + HIDDEN_DIM))
    w1 = [[random.gauss(0, std1) for _ in range(HIDDEN_DIM)] for _ in range(STATE_DIM)]
    b1 = [0.0] * HIDDEN_DIM

    std2 = math.sqrt(2.0 / (HIDDEN_DIM + ACTION_DIM))
    w2 = [[random.gauss(0, std2) for _ in range(ACTION_DIM)] for _ in range(HIDDEN_DIM)]
    b2 = [0.0] * ACTION_DIM

    return {"w1": w1, "b1": b1, "w2": w2, "b2": b2}


def forward(
    state: list[float],
    params: dict,
    mask: list[bool],
) -> tuple[list[float], list[float]]:
    """Forward pass: state -> masked action probabilities.

    Math:
      h_j = ReLU( Σ_i s_i * W1[i,j] + b1_j )          -- hidden layer
      z_k = Σ_j h_j * W2[j,k] + b2_k                   -- raw logits
      z_k = -inf  if mask[k] is False                    -- action masking
      π(a_k|s) = exp(z_k - max(z)) / Σ_k' exp(z_k' - max(z))  -- softmax

    Action masking sets logits to -inf for invalid actions before softmax,
    ensuring zero probability for masked actions. This is mathematically
    equivalent to softmax over only the valid actions.

    Returns (probs, hidden) -- hidden cached for backward pass.
    """
    w1, b1, w2, b2 = params["w1"], params["b1"], params["w2"], params["b2"]

    # Hidden layer with ReLU activation
    hidden = [0.0] * HIDDEN_DIM
    for j in range(HIDDEN_DIM):
        val = b1[j]
        for i in range(STATE_DIM):
            val += state[i] * w1[i][j]
        hidden[j] = max(0.0, val)

    # Output logits with masking
    logits = [0.0] * ACTION_DIM
    for k in range(ACTION_DIM):
        val = b2[k]
        for j in range(HIDDEN_DIM):
            val += hidden[j] * w2[j][k]
        logits[k] = val if mask[k] else -1e9  # -inf for masked actions

    # Numerically stable softmax
    max_l = max(logits)
    exps = [math.exp(l - max_l) for l in logits]
    total = sum(exps)
    probs = [e / total for e in exps]

    return probs, hidden


def sample_action(probs: list[float]) -> int:
    """Inverse CDF sampling from categorical distribution."""
    u = random.random()
    cdf = 0.0
    for i, p in enumerate(probs):
        cdf += p
        if u <= cdf:
            return i
    return len(probs) - 1


# === REINFORCE TRAINING ===

# REINFORCE gradient for softmax output:
#   ∂ log π(a|s) / ∂ z_k = 𝟙(k=a) - π(k|s)
#
# This elegant identity ("softmax trick") makes policy gradients over discrete
# actions trivially cheap to compute. For the chosen action a: gradient = (1 - prob),
# pushing its logit up. For all others: gradient = (-prob), pushing them down.
#
# With action masking, we only compute gradients for unmasked actions. Masked
# actions have π ≈ 0, so their contribution vanishes automatically.
#
# Full backward pass (math-to-code mapping):
#   ∂L/∂z_k = advantage * (𝟙(k=a) - π_k)        -- output gradient
#   ∂L/∂W2[j,k] = h_j * ∂L/∂z_k                  -- chain rule: W2
#   ∂L/∂h_j = Σ_k W2[j,k] * ∂L/∂z_k             -- backprop to hidden
#   ∂L/∂h_j *= 𝟙(h_j > 0)                        -- ReLU gradient gate
#   ∂L/∂W1[i,j] = s_i * ∂L/∂h_j                  -- chain rule: W1

def compute_gradients(
    trajectories: list[dict],
    params: dict,
    baseline: float,
) -> dict:
    """REINFORCE gradient computation over a batch of trajectories."""
    gw1 = [[0.0] * HIDDEN_DIM for _ in range(STATE_DIM)]
    gb1 = [0.0] * HIDDEN_DIM
    gw2 = [[0.0] * ACTION_DIM for _ in range(HIDDEN_DIM)]
    gb2 = [0.0] * ACTION_DIM
    n = 0

    for traj in trajectories:
        advantage = traj["reward"] - baseline

        for sd in traj["steps"]:
            state, action, probs, hidden, mask = (
                sd["state"], sd["action"], sd["probs"], sd["hidden"], sd["mask"]
            )

            # Softmax gradient: ∂log π(a)/∂z_k = 𝟙(k=a) - π_k
            d_logits = [0.0] * ACTION_DIM
            for k in range(ACTION_DIM):
                if not mask[k]:
                    continue  # skip masked actions entirely
                indicator = 1.0 if k == action else 0.0
                d_logits[k] = advantage * (indicator - probs[k])
                # Entropy bonus: -coeff * (log π_k + 1) * (𝟙 - π_k)
                pk = max(probs[k], 1e-10)
                d_logits[k] += ENTROPY_COEFF * (indicator - probs[k]) * (-math.log(pk) - 1)

            # Backprop through W2
            for k in range(ACTION_DIM):
                gb2[k] += d_logits[k]
                for j in range(HIDDEN_DIM):
                    gw2[j][k] += hidden[j] * d_logits[k]

            # Backprop through hidden (ReLU gated)
            d_hidden = [0.0] * HIDDEN_DIM
            for j in range(HIDDEN_DIM):
                for k in range(ACTION_DIM):
                    d_hidden[j] += params["w2"][j][k] * d_logits[k]
                if hidden[j] <= 0:
                    d_hidden[j] = 0.0

            # Backprop through W1
            for j in range(HIDDEN_DIM):
                gb1[j] += d_hidden[j]
                for i in range(STATE_DIM):
                    gw1[i][j] += state[i] * d_hidden[j]

            n += 1

    if n > 0:
        s = 1.0 / n
        for i in range(STATE_DIM):
            for j in range(HIDDEN_DIM):
                gw1[i][j] *= s
        for j in range(HIDDEN_DIM):
            gb1[j] *= s
            for k in range(ACTION_DIM):
                gw2[j][k] *= s
        for k in range(ACTION_DIM):
            gb2[k] *= s

    return {"w1": gw1, "b1": gb1, "w2": gw2, "b2": gb2}


def update_params(params: dict, grads: dict, lr: float) -> None:
    """Gradient ascent (REINFORCE maximizes expected reward) with clipping.

    Clipping caps extreme gradient updates caused by REINFORCE's high variance.
    Without it, a single outlier trajectory can destabilize the policy.
    """
    clip = 2.0
    for key in ("w1", "w2"):
        for i in range(len(params[key])):
            for j in range(len(params[key][i])):
                g = max(-clip, min(clip, grads[key][i][j]))
                params[key][i][j] += lr * g
    for key in ("b1", "b2"):
        for i in range(len(params[key])):
            g = max(-clip, min(clip, grads[key][i]))
            params[key][i] += lr * g


# === REWARD FUNCTION ===

# Shaped rewards provide a denser learning signal than sparse outcome-only reward.
# This mirrors "process reward models" (Lightman et al., 2023): rewarding good
# reasoning steps, not just correct conclusions.
#
# The reward structure encodes the ReAct insight: good episodes follow the
# pattern lookup -> lookup -> compute -> correct answer. Each component of
# this pattern gets partial credit.

def compute_reward(
    question: Question,
    observations: list[dict],
    final_answer: str | float | None,
) -> float:
    """Score a complete episode. Shaped reward for intermediate steps + outcome."""
    reward = 0.0
    looked_up: set[str] = set()
    target_entities = set(question["entities"])

    for obs in observations:
        if obs["tool"] == "lookup":
            ent = obs.get("entity", "")
            if ent in target_entities and ent not in looked_up:
                reward += 0.15  # correct entity, first time
            looked_up.add(ent)
        elif obs["tool"] in ("compare", "calculate"):
            if obs.get("result") != "missing_values":
                reward += 0.05  # valid computation attempted

    # Final answer: the largest reward component
    correct = question["answer"]
    atype = question["answer_type"]

    if final_answer is not None:
        if atype in ("comparison", "comparison_min"):
            if isinstance(final_answer, str) and final_answer.lower() == str(correct).lower():
                reward += 0.60
        else:
            try:
                if float(correct) != 0:
                    ratio = float(final_answer) / float(correct)
                    if 0.9 <= ratio <= 1.1:
                        reward += 0.60
            except (ValueError, TypeError, ZeroDivisionError):
                pass

    return reward


# === ANSWER DERIVATION ===

def derive_answer(
    question: Question,
    observations: list[dict],
    retrieved_values: dict[str, float | int | None],
) -> str | float | None:
    """Synthesize the final answer from the observation history.

    For comparison: map compare result to entity name.
    For arithmetic: return the compute result.
    Fallback: if values were retrieved but wrong compute was used,
    attempt correct derivation from raw values.
    """
    entities = question["entities"]
    atype = question["answer_type"]

    # Explicit computation results take priority
    for obs in reversed(observations):
        if obs["tool"] == "compare" and obs.get("result") not in ("missing_values", None):
            if obs["result"] == "first_greater":
                return entities[1] if atype == "comparison_min" else entities[0]
            elif obs["result"] == "second_greater":
                return entities[0] if atype == "comparison_min" else entities[1]
            return entities[0]  # equal

        if obs["tool"] == "calculate" and isinstance(obs.get("value"), (int, float)):
            val = obs["value"]
            return round(val, 2) if isinstance(val, float) else val

    # Fallback: derive from raw retrieved values if both present.
    # This rescues episodes where the agent looked up both entities but chose
    # the wrong compute action (e.g., sum instead of compare).
    v1 = retrieved_values.get(entities[0])
    v2 = retrieved_values.get(entities[1])

    if v1 is not None and v2 is not None:
        fv1, fv2 = float(v1), float(v2)
        if atype == "comparison":
            return entities[0] if fv1 > fv2 else entities[1]
        elif atype == "comparison_min":
            return entities[0] if fv1 < fv2 else entities[1]
        elif atype == "sum":
            return round(fv1 + fv2, 2)
        elif atype == "difference":
            return round(abs(fv1 - fv2), 2)
        elif atype == "ratio":
            return round(fv1 / fv2, 2) if fv2 != 0 else None

    return None


# === EPISODE EXECUTION ===

def run_episode(
    question: Question,
    params: dict,
    greedy: bool = False,
) -> dict:
    """Execute one full ReAct episode: Thought -> Action -> Observation loop.

    The state machine at each step:
      THOUGHT: Encode state -> policy forward pass -> action distribution.
               The hidden activations ARE the "thought" -- they encode the
               agent's reasoning about what to do next.
      ACTION:  Sample (or greedy-select) from masked action distribution.
               Masking ensures only valid actions are considered.
      OBSERVE: Execute the tool, update observation history and state.
      (Repeat for MAX_STEPS or until early termination.)

    This mirrors text-based ReAct exactly. Neural activations replace natural
    language thoughts; action masking replaces the LLM's learned grammar of
    valid tool invocations.
    """
    observations: list[dict] = []
    retrieved_values: dict[str, float | int | None] = {}
    step_records: list[dict] = []
    trace: list[str] = []
    entities = question["entities"]

    for step in range(MAX_STEPS):
        state = encode_state(question, step, retrieved_values)
        mask = get_action_mask(question, retrieved_values)
        probs, hidden = forward(state, params, mask)

        if greedy:
            action = probs.index(max(probs))
        else:
            action = sample_action(probs)

        step_records.append({
            "state": state, "action": action, "probs": probs,
            "hidden": hidden, "mask": mask,
        })

        obs = execute_action(action, question, retrieved_values)
        observations.append(obs)

        # Update retrieved values
        if obs["tool"] == "lookup" and obs.get("value") is not None:
            retrieved_values[obs["entity"]] = obs["value"]

        # Build trace
        trace.append(f"  Step {step + 1}:")
        trace.append(f"    Thought:  {_thought_text(action, probs, mask)}")
        trace.append(f"    Action:   {_action_text(action, obs, entities)}")
        trace.append(f"    Observe:  {_observe_text(obs)}")

        # Early termination: both looked up + computation done
        both = all(retrieved_values.get(e) is not None for e in entities)
        computed = any(o["tool"] in ("compare", "calculate") for o in observations
                       if o.get("result") not in ("missing_values", None))
        if both and computed:
            break

    final_answer = derive_answer(question, observations, retrieved_values)
    reward = compute_reward(question, observations, final_answer)
    trace.append(f"    Answer:   {final_answer}")

    return {
        "steps": step_records,
        "observations": observations,
        "final_answer": final_answer,
        "reward": reward,
        "trace": trace,
    }


def _thought_text(action: int, probs: list[float], mask: list[bool]) -> str:
    """Generate human-readable thought from action distribution."""
    valid = [i for i, m in enumerate(mask) if m]
    conf = probs[action]
    if action <= 1:
        phase = "Gathering info"
        remaining = sum(1 for i in (0, 1) if mask[i])
        return f"{phase} ({remaining} lookup(s) needed, conf: {conf:.2f})"
    else:
        op_names = {2: "compare", 3: "sum", 4: "difference", 5: "ratio", 6: "ratio"}
        op = op_names.get(action, "compute")
        return f"Values retrieved, computing {op} (conf: {conf:.2f})"


def _action_text(action: int, obs: dict, entities: list[str]) -> str:
    if obs["tool"] == "lookup":
        return f"lookup({obs['entity']})"
    elif obs["tool"] == "compare":
        return f"compare({obs.get('a', '?')}, {obs.get('b', '?')})"
    elif obs["tool"] == "calculate":
        return f"calculate({obs.get('a', '?')} {obs.get('op', '?')} {obs.get('b', '?')})"
    return f"action_{action}"


def _observe_text(obs: dict) -> str:
    if obs["tool"] == "lookup":
        return f"{obs['entity']} -> {obs.get('value', '(not found)')}"
    elif obs["tool"] == "compare":
        mapping = {"first_greater": "first > second",
                    "second_greater": "second > first", "equal": "equal"}
        return mapping.get(obs["result"], str(obs["result"]))
    elif obs["tool"] == "calculate":
        return f"= {obs.get('result', '?')}"
    return str(obs.get("result", "?"))


# === RANDOM BASELINE ===

def run_random_episode(question: Question) -> dict:
    """Episode with uniformly random (but mask-respecting) action selection.

    Even with masking, random selection rarely produces correct answers because
    it must randomly choose the right compute operation (1 out of 5 for arithmetic
    questions). The trained agent learns which operation matches each question type.
    """
    observations: list[dict] = []
    retrieved_values: dict[str, float | int | None] = {}
    trace: list[str] = []

    for step in range(MAX_STEPS):
        mask = get_action_mask(question, retrieved_values)
        valid = [i for i, m in enumerate(mask) if m]
        action = random.choice(valid)

        obs = execute_action(action, question, retrieved_values)
        observations.append(obs)
        if obs["tool"] == "lookup" and obs.get("value") is not None:
            retrieved_values[obs["entity"]] = obs["value"]

        trace.append(f"  Step {step + 1}: {ACTION_NAMES[action]} (random) -> {_observe_text(obs)}")

    final_answer = derive_answer(question, observations, retrieved_values)
    reward = compute_reward(question, observations, final_answer)
    trace.append(f"  Answer: {final_answer}")
    return {"final_answer": final_answer, "reward": reward, "trace": trace}


# === TRAINING LOOP ===

def train() -> dict:
    """Train the ReAct policy with REINFORCE + baseline.

    Training procedure:
      1. Run all questions through the current policy (collect trajectories)
      2. Compute advantage: R_episode - EMA_baseline
      3. Compute ∇log π(a|s) * advantage for each step
      4. Gradient ascent on policy parameters
      5. Update EMA baseline

    The EMA baseline is essential for REINFORCE variance reduction:
    without it, all actions get reinforced (just at different magnitudes).
    Subtracting the baseline centers advantages around zero, so only
    above-average trajectories get positive gradient updates.
    """
    print("=" * 65)
    print("TRAINING ReAct AGENT")
    print("=" * 65)
    print(f"  State dim:    {STATE_DIM}")
    print(f"  Hidden dim:   {HIDDEN_DIM}")
    print(f"  Action space: {ACTION_DIM} (2 lookups + 5 compute ops, masked)")
    print(f"  Questions:    {len(QUESTIONS)}")
    print(f"  Epochs:       {NUM_EPOCHS}")
    print(f"  Max steps:    {MAX_STEPS} per question")
    print()

    params = init_policy()
    baseline = 0.0
    t0 = time.time()
    acc_hist: list[float] = []
    rew_hist: list[float] = []

    for epoch in range(NUM_EPOCHS):
        trajs: list[dict] = []
        rewards: list[float] = []
        correct = 0

        for q in QUESTIONS:
            traj = run_episode(q, params, greedy=False)
            trajs.append(traj)
            rewards.append(traj["reward"])
            if _is_correct(q, traj["final_answer"]):
                correct += 1

        avg_r = sum(rewards) / len(rewards)
        acc = correct / len(QUESTIONS)
        acc_hist.append(acc)
        rew_hist.append(avg_r)

        baseline = BASELINE_DECAY * baseline + (1 - BASELINE_DECAY) * avg_r

        grads = compute_gradients(trajs, params, baseline)
        lr = LEARNING_RATE * (1.0 - 0.7 * epoch / NUM_EPOCHS)
        update_params(params, grads, lr)

        if (epoch + 1) % 50 == 0 or epoch == 0:
            elapsed = time.time() - t0
            print(f"  Epoch {epoch + 1:3d}/{NUM_EPOCHS}  |  "
                  f"Reward: {avg_r:+.3f}  |  "
                  f"Accuracy: {acc:5.1%}  |  "
                  f"Baseline: {baseline:+.3f}  |  "
                  f"LR: {lr:.4f}  |  "
                  f"{elapsed:.1f}s")

    total_time = time.time() - t0
    print(f"\n  Training complete in {total_time:.1f}s")
    print(f"  Initial accuracy:  {acc_hist[0]:.1%}")
    print(f"  Final accuracy:    {acc_hist[-1]:.1%}")
    print(f"  Peak accuracy:     {max(acc_hist):.1%}")
    print(f"  Initial reward:    {rew_hist[0]:+.3f}")
    print(f"  Final reward:      {rew_hist[-1]:+.3f}")

    return params


def _is_correct(question: Question, answer: str | float | None) -> bool:
    """Check if the agent's answer matches expected."""
    if answer is None:
        return False
    correct = question["answer"]
    atype = question["answer_type"]
    if atype in ("comparison", "comparison_min"):
        return isinstance(answer, str) and answer.lower() == str(correct).lower()
    try:
        if float(correct) != 0:
            ratio = float(answer) / float(correct)
            return 0.9 <= ratio <= 1.1
    except (ValueError, TypeError, ZeroDivisionError):
        pass
    return False


# === INFERENCE AND DEMO ===

def demo(params: dict) -> None:
    """Evaluate trained agent vs random baseline with full reasoning traces.

    Prints the complete Thought -> Action -> Observation chain for each question,
    making every decision visible and auditable. This transparency is a core
    advantage of ReAct over opaque end-to-end models.
    """
    # --- Random Baseline ---
    print("\n" + "=" * 65)
    print("RANDOM BASELINE (with action masking)")
    print("=" * 65)

    random_correct = 0
    random_reward = 0.0
    # Run multiple random trials and average (reduces variance in baseline estimate)
    n_random_trials = 5
    for _ in range(n_random_trials):
        rc, rr = 0, 0.0
        for q in QUESTIONS:
            result = run_random_episode(q)
            if _is_correct(q, result["final_answer"]):
                rc += 1
            rr += result["reward"]
        random_correct += rc
        random_reward += rr
    random_acc = random_correct / (len(QUESTIONS) * n_random_trials)
    random_avg_r = random_reward / (len(QUESTIONS) * n_random_trials)
    print(f"\n  Accuracy (avg of {n_random_trials} runs): {random_acc:.1%}  |  "
          f"Avg reward: {random_avg_r:+.3f}")

    # --- Trained Agent ---
    print("\n" + "=" * 65)
    print("TRAINED ReAct AGENT - FULL REASONING TRACES")
    print("=" * 65)

    trained_correct = 0
    trained_reward = 0.0

    for i, q in enumerate(QUESTIONS):
        print(f"\nQ{i + 1}: {q['text']}")
        print(f"  Expected: {q['answer']}")

        result = run_episode(q, params, greedy=True)
        for line in result["trace"]:
            print(line)

        ok = _is_correct(q, result["final_answer"])
        print(f"  [{'CORRECT' if ok else 'WRONG'}]  reward: {result['reward']:+.3f}")

        if ok:
            trained_correct += 1
        trained_reward += result["reward"]

    trained_acc = trained_correct / len(QUESTIONS)
    trained_avg_r = trained_reward / len(QUESTIONS)

    # --- Summary ---
    print("\n" + "=" * 65)
    print("RESULTS SUMMARY")
    print("=" * 65)
    print(f"  Random agent:   {random_acc:5.1%} accuracy  |  {random_avg_r:+.3f} avg reward")
    print(f"  Trained agent:  {trained_acc:5.1%} accuracy  |  {trained_avg_r:+.3f} avg reward")
    print(f"  Improvement:    {trained_acc - random_acc:+.1%} accuracy gain")

    # --- Educational Summary ---
    print("\n" + "=" * 65)
    print("KEY INSIGHTS")
    print("=" * 65)
    print("""
  ReAct vs Chain-of-Thought (CoT):
    CoT generates reasoning as ungrounded text -- it can "hallucinate" facts.
    ReAct interleaves reasoning with tool calls that return real observations.
    Observations anchor each reasoning step in ground truth, reducing drift.

  The State Machine Formalism:
    THOUGHT: Policy network processes state -> decides what to do next.
    ACTION:  Execute a tool call (lookup, calculate, compare).
    OBSERVE: Record the tool's output, update the state representation.
    Every decision is fully observable and auditable.

  Why Interleaving Matters:
    Pure reasoning (no tools) drifts from reality. Pure action (no reasoning)
    is aimless trial-and-error. ReAct's interleaving lets each observation
    refine the next thought, creating a self-correcting reasoning chain.

  Action Masking:
    Constraining the action space at each step (only valid tools are selectable)
    is equivalent to structured decoding in LLM agents. It prevents wasted
    exploration on impossible actions and focuses learning on the real decision:
    WHICH entity to look up, and WHICH operation to apply.

  The Training Signal:
    REINFORCE learns from outcome: correct final answers get positive reward,
    shaped by intermediate credit for correct tool usage. This mirrors
    "process reward models" that reward good reasoning steps, not just
    correct conclusions.

  Connection to Modern LLM Agents:
    ChatGPT plugins, function calling, and LangChain agents implement this
    same Thought -> Action -> Observation loop. They use an LLM as the
    policy (generating thoughts as text); we use a trained neural network.
    The algorithmic structure -- structured interleaving of reasoning and
    grounded observation -- is identical.
""")


# === MAIN ===

if __name__ == "__main__":
    trained_params = train()
    demo(trained_params)
