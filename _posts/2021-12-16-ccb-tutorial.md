---
title: 'Multi-Slot Contextual Bandit Recommender Systems Using Vowpal Wabbit'
date: 2021-12-16
permalink: /posts/2021/12/ccb-tutorial/
tags:
  - multi-armed bandits
  - recommender system
  - reinforcement learning
---

[See on Medium](https://medium.com/@zaid-g/multi-slot-contextual-bandit-selection-using-vowpal-wabbit-ddfe391173a) Computationally efficient contextual bandit multi-action selection using Vowpal Wabbit tutorial.

-------------------------------------------------------

Contextual Bandits are a class of online learning algorithms that model an agent that learns to act optimally by efficiently acquiring new knowledge and exploiting it. They are used in a variety of settings (e.g. clinical trials, recommender systems, finance). Typically, contextual bandits are used for selecting a single action at each round based on the observed contextual features. For example, in a news article website, a contextual bandit may be used to select a single news article from a list of candidate articles to highlight at the top of the page each time a user visits. The selection process aims to maximize the long term reward (e.g. engagement of the user).

However, what if we want to use a CB to select multiple actions in each round? What if there are multiple _slots_ for which to select an item from the set of candidate items at each round? Or in our news article example, what if there are 6 slots on the front page and we would like to fill each with any of 100 candidate news articles each time a user visits the page? Reframing the process, a unique action in this case may be defined as selecting all 6 items for each user visit, but the size of the set of unique actions becomes extremely large: almost 1 trillion (Permute(100, 6)).


| ![homepage.jpg](/images/homepage.png) | 
|:--:| 
| *A typical news website homepage that displays top articles in the alotted slots* |


Several methods exist to tackle this problem, such as various Contextual Combinatorial Bandit approaches. But a simpler way of approaching it is sequentially sampling an action for each slot, for each round. For example, a contextual bandit samples one out of the 100 articles to fill the first slot, 1 out of the 99 remaining articles for the second slot, etc.., and reward is collected and learned from for each slot individually. This makes the problem computationally feasible. 

However, this approach breaks the bandit's i.i.d. assumption: selecting an action at a specific timestep affects the set of available actions for future timesteps (within the same round). Care needs to be taken for this to not be a problem. Specifically, we must begin sampling actions for the slots that have the highest reward potential first. This is because e.g. if we begin by sampling a news article for a slot that is at the bottom of the page, then even if the selected news article is of high quality, the reward will be low since it is unlikely to be looked at. And that high quality article becomes surreptitiously unavailable to the bandit for slots at the top of the page, since it was already selected in the round. If the fullest action sets are made available for the best slots, the long term cumulative reward is going to be larger.

In this article, we'll show how to to use [Vowpal Wabbit's](https://vowpalwabbit.org/) [Conditional Contextual Bandit](https://github.com/VowpalWabbit/vowpal_wabbit/wiki/Conditional-Contextual-Bandit) module, which allows us to implement this approach in Python. This library supports a dynamically changing action set (and set size) for each round and for each slot, contextual features for each action, contextual features for each slot, and a namespace for shared contextual features (for all slots and actions). The algorithmic approach to recommendation detailed here is similar to the one that powers the Microsoft Azure Personalizer product.

## Python simulation using Vowpal Wabbit

[Vowpal wabbit](https://vowpalwabbit.org/) is a fast Online Interactive Learning library that offers several contextual bandit implementations. The [Conditional Contextual Bandit](https://github.com/VowpalWabbit/vowpal_wabbit/wiki/Conditional-Contextual-Bandit) extension is a wrapper over these implementations for the setting where there are multiple slots for which an action can be chosen.

** Full code available [here](https://github.com/zaid-g/ccb_tutorial/blob/main/ccb_tutorial.py).

Module version details:
* vowpalwabbit==8.11.0
* python 3.9.6

First, let's write our imports and define a CCB VW model:

```
from vowpalwabbit import pyvw
import random
from pprint import pprint as P
import numpy as np
from matplotlib import pyplot as plt

seed = random.randint(0, 100000)

m = pyvw.vw(f"--ccb_explore_adf  --cover 5 -q :: --random_seed {seed}") # "-q ::" needed to learn properly
```

We want to simulate an environment in which every time a user visits the page, 6 news articles are selected from 10 candidate articles to be displayed. We also want to challenge the algorithm by encoding in the users' behavior to ignore any article displayed in any slot after the 3rd to see how that affects the performance. Let's first define two user profiles and a reward function for each. The user profile features will be in the shared namespace (since they apply to all actions and slots). 

```
# We will choose 1 of two users at random for each round with equal probability.
# Each user has 3 features: the user's unique ID (categorical), the time since
# they last logged on (float), and whether or not they are a subscriber (categorical).
# Note that, in VW format, ordinal features (e.g. last_opened:[Float]) need the colon
# separator between the feature name its corresponding value. The colon should not be
# used for categorical features.
user_1_ccb_string = "ccb shared | userid=1 last_opened:0.5 subscriber=true\n"
user_2_ccb_string = "ccb shared | userid=2 last_opened:1.6 subscriber=false\n"

# We next define the reward function based on the selected action (article) for each user.
# For simplicity, we return the reward directly. These rewards could be interpreted
# as the level of engagement of the user, for example the time the user spent reading it.
# For example, rewards of 1.7 and 0.2 could mean that the user spent 1700s and 200s reading
# the article
user_1_reward_dict = {
    "article_1": 0.2,
    "article_2": 0.3,
    "article_3": 0,
    "article_4": 0,
    "article_5": 0,
    "article_6": 0.9,
    "article_7": 1.7,
    "article_8": 0,
    "article_9": 0,
    "article_10": 0,
}
user_2_reward_dict = {
    "article_1": 0.1,
    "article_2": 0.4,
    "article_3": 2.9,
    "article_4": 0,
    "article_5": 0,
    "article_6": 1,
    "article_7": 0.5,
    "article_8": 0,
    "article_9": 0,
    "article_10": 1,
}


def sample_user():
    # choose one of the two users at random
    if random.random() < 0.5:
        return user_1_ccb_string, user_1_reward_dict
    else:
        return user_2_ccb_string, user_2_reward_dict


# simulated user will ignore recommendations after this slot index (reward = 0)
ignore_after_index = 2


def simulate_reward(
    slot_index, chosen_action_features, reward_dict, ignore_after_index
):
    if slot_index > ignore_after_index:
        return 0
    return reward_dict[chosen_action_features]
```

Following the [CCB string format](https://github.com/VowpalWabbit/vowpal_wabbit/wiki/Conditional-Contextual-Bandit#vw-text-format), let's define the action and slot sets. We will shuffle the action strings for each round and format them in CCB format, which is what the `get_actions_ccb_string` function is for, to simulate a more realistic environment.

```
# define slots in VW CCB format
num_slots = 6
slots_ccb_string = "ccb slot  | \n" * num_slots


action_strings = (
    "article_1",
    "article_2",
    "article_3",
    "article_4",
    "article_5",
    "article_6",
    "article_7",
    "article_8",
    "article_9",
    "article_10",
)  # different actions defined by their features in vw string format


def get_actions_ccb_string(action_strings):
    action_strings = random.sample(action_strings, len(action_strings))
    actions_ccb_string = ""
    for action_string in action_strings:
        actions_ccb_string += f"ccb action | {action_string}\n"
    return actions_ccb_string
```

By defining the users' reward functions, actions, and slots, we also can compute the maximum average reward possible.


```
# since we know how the users respond to the actions and # of slots,
# we know how best we can do (on average)
best_policy_average_reward = (
    sum(
        sorted(list(user_1_reward_dict.values()), reverse=True)[
            0 : min(ignore_after_index + 1, num_slots)
        ]
    )
    + sum(
        sorted(list(user_2_reward_dict.values()), reverse=True)[
            0 : min(ignore_after_index + 1, num_slots)
        ]
    )
) / 2
```

We need to define two more functions that will come in handy for when we create the CCB update strings that the model will learn from after each round:

```
def to_ccb_slot_format_result(result):
    """converts the chosen action, cost, and probability tuples for each slot into
    a string for creating the VW CCB update string.
    """
    result = (
        str(
            [
                str(x)
                .replace(" ", "")
                .replace("(", "")
                .replace(")", "")
                .replace("[", "")
                .replace("]", "")
                .replace(",", ":")
                for x in result
            ]
        )
        .replace("[", "")
        .replace("]", "")
        .replace(" ", "")
        .replace("'", "")
    )
    return result


def replace_nth_string_occurance(s, source, target, n):
    """Adds the action, cost, and probability string at the right slot index of the predict
    string to create the update string in CCB Input Format.

    The final output for each slot will be of the form
    "ccb slot [<chosen_action>:<cost>:<probability>,<action>:<probability>,...] | ..."
    as is specified in the CCB Input Format (here we assume that there are no slot
    specific action restrictions besides the available actions, so it won't include the
    [action_ids_to_include,...] part shown in the documentation).

    Args:
        s (Str): source string
        source (Str): string occurence to be replaced
        target (Str): string occurrence to be replaced with
        n (Int): index of string occurence to be replaced

    Returns: s with nth string (source) occurence replaced with target

    """
    inds = [
        i for i in range(len(s) - len(source) + 1) if s[i : i + len(source)] == source
    ]
    if len(inds) < n:
        return  # or maybe raise an error
    s = list(s)  # can't assign to string slices. So, let's listify
    s[
        inds[n - 1] : inds[n - 1] + len(source)
    ] = target  # do n-1 because we start from the first occurrence of the string, not the 0-th
    return "".join(s)
```

Finally, let's run the simulation! We will run 500 rounds, so our model will choose 500*6=3000 actions in total. We'll first run a random policy (a policy that chooses actions uniformly at random) to see how it does:

```
# simulation
num_rounds = 500  # we'll simulate 500 rounds

# random policy simulation
round_results_random_policy = []
round_rewards_random_policy = []
for _ in range(num_rounds):
    # sample user
    _, reward_dict = sample_user()
    # choose num_slots actions (without replacement)
    round_result_random_policy = random.sample(action_strings, num_slots)
    # get simulated reward based on chosen actions of random policy
    round_reward_random_policy = sum(
        [
            simulate_reward(
                i, round_result_random_policy[i], reward_dict, ignore_after_index
            )
            for i in range(num_slots)
        ]
    )
    # store result and reward
    round_rewards_random_policy.append(round_reward_random_policy)
    round_results_random_policy.append(round_result_random_policy)
```

And now we'll run the simulation using the bandit:

```
# CCB simulation
round_results_ccb = []
round_rewards_ccb = []
for _ in range(num_rounds):
    # shuffle actions list in VW CCB format to mimic realistic setting of changing actions set
    actions_ccb_string = get_actions_ccb_string(action_strings)
    user_ccb_string, reward_dict = sample_user()
    input_ccb_string = user_ccb_string + actions_ccb_string + slots_ccb_string
    # call model for choosing action for each slot
    round_result_ccb = m.predict(input_ccb_string)
    ccb_round_reward = 0
    # initialize update string
    update_ccb_string = input_ccb_string
    for slot_index in range(len(round_result_ccb)):
        # get simulated reward based on chosen action of bandit model
        chosen_action_index = round_result_ccb[slot_index][0][0]
        chosen_action_features = actions_ccb_string.split("\n")[
            chosen_action_index
        ].split("| ", 1)[1]
        reward = simulate_reward(
            slot_index, chosen_action_features, reward_dict, ignore_after_index
        )
        ccb_round_reward += reward
        # incorporate reward and actions selected into update string
        round_result_ccb[slot_index][0] = (
            round_result_ccb[slot_index][0][0],
            -reward,  # cost = -reward
            round_result_ccb[slot_index][0][1],
        )
        update_ccb_string = replace_nth_string_occurance(
            update_ccb_string,
            "ccb slot",
            "ccb slot " + to_ccb_slot_format_result(round_result_ccb[slot_index]),
            slot_index + 1,
        )
    # update model using fully formulated update string for the round
    m.learn(update_ccb_string)
    # store result and reward
    round_results_ccb.append(round_result_ccb)
    round_rewards_ccb.append(ccb_round_reward)

```

Since we know the best possible actions that can be chosen for each round (and the best average reward), we can compute and plot the _regret_, which is the difference between the expected reward from the optimal policy and the actual collected reward for each round.

```
# compute regret
round_regrets_ccb = best_policy_average_reward - np.array(round_rewards_ccb)
round_regrets_random_policy = best_policy_average_reward - np.array(
    round_rewards_random_policy
)
# plot rolling average regret
roll = 10
rolling_round_regrets_ccb = np.convolve(
    round_regrets_ccb, np.ones(roll) / roll, mode="valid"
)
rolling_round_regrets_random_policy = np.convolve(
    round_regrets_random_policy, np.ones(roll) / roll, mode="valid"
)
print(np.mean(round_regrets_ccb))
plt.plot(list(range(len(rolling_round_regrets_ccb))), rolling_round_regrets_ccb)
plt.plot(
    list(range(len(rolling_round_regrets_random_policy))),
    rolling_round_regrets_random_policy,
    label="random policy",
)
plt.plot(
    list(range(len(rolling_round_regrets_ccb))),
    [0] * len(rolling_round_regrets_ccb),
    label="ccb",
)
plt.xlabel("round")
plt.ylabel(f"rolling average regret (N={roll})")
plt.ylim(-1, best_policy_average_reward)
plt.title(
    f"Rolling regret for 10 candidates, 6 slots, user ignores all slots after #{ignore_after_index + 1}"
)
plt.legend(loc="upper left")
if ignore_after_index == 2:
    plt.savefig("sim_0")
elif ignore_after_index == 5:
    plt.savefig("sim_1")
plt.show()
```

### Results


![](/images/sim_0.png)

We can see that the CCB policy's *regret* is slowly trending downwards towards 0, which means that the strategy is updating towards the optimal one.

However, the convergence is not occuring as fast as expected, this is because we coded the users' reward functions to always return zero regardless of the action chosen for the 4th, 5th, and 6th slots. This is meant to emulate a realistic setting in which users don't interact with all the recommendations, just the top few. Removing this behavior by changing the `ignore_after_index` variable value to >= 5 and rerunning the simulation results in much quicker convergence, as seen in the next plot:

Note that the reason why the regret is less than zero sometimes is the difference in the reward functions of the two users and the random selection of the users for each round.

![](/images/sim_1.png)

### Performance

Running the entire simulation took only 0.744s on my machine, which means that the time to predict **and** update is only about 1 millisecond per round. So the library is very performant and suitable for production environments.

## Conclusion

The sample code above can be utilized for implementing a computationally efficient Contextual Bandit approach for multi-action/multi-item selection in each round. The benefits of using an explore/exploit approach such as a higher diversity of recommendations and better handling of sparse data/cold starts can be extended for this setting.
