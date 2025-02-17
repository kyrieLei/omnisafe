from datetime import datetime
from functional import seq
import numpy as np
import time
import torch
import torch.nn.functional as F
from torch.optim import Adam

from omnisafe.common.safe_explorer.core.config import Config
from omnisafe.common.safe_explorer.core.replay_buffer import ReplayBuffer

from omnisafe.common.safe_explorer.safety_layer.constraint_model import ConstraintModel
from omnisafe.common.safe_explorer.utils.list import for_each


class SafetyLayer:
    def __init__(self, env, buf):
        self._env = env
        self._buf = buf
        self._config = Config.get().safety_layer.trainer
        self._initialize_constraint_models()
        self._num_constraints=1

        self._replay_buffer = ReplayBuffer(self._config.replay_buffer_size)

        self._train_global_step = 0
        self._eval_global_step = 0

        if self._config.use_gpu:
            self._cuda()

    def _as_tensor(self, ndarray, requires_grad=False):
        tensor = torch.Tensor(ndarray)
        tensor.requires_grad = requires_grad
        return tensor

    def _cuda(self):
        for_each(lambda x: x.cuda(), self._models)

    def _eval_mode(self):
        for_each(lambda x: x.eval(), self._models)

    def _train_mode(self):
        for_each(lambda x: x.train(), self._models)

    def _initialize_constraint_models(self):
        self._models = [ConstraintModel(self._env.observation_space.shape[0],
                                        self._env.action_space.shape[0]) \
                        for _ in range(1)]
        self._optimizers = [Adam(x.parameters(), lr=self._config.lr) for x in self._models]

    def _sample_steps(self, num_steps):
        episode_length = 0

        for step in range(num_steps):

            data = self._buf.sample_batch()
            act, done, observation_next,obs = (
                data['act'],
                data['done'],
                data['next_obs'],
                data['obs']
            )
            self._replay_buffer.add({
                "action": act,
                "observation": obs,
                "c": 10,
                "c_next": 10
            })

            observation = observation_next
            episode_length += 1

            if done or (episode_length == self._config.max_episode_length):
                observation = self._env.reset()
                episode_length = 0

    def _evaluate_batch(self, batch):
        observation = self._as_tensor(batch["obs"])
        action = self._as_tensor(batch["act"])
        c = 10
        c_next = 10

        gs = [x(observation) for x in self._models]

        c_next_predicted = [c[:, i] + \
                            torch.bmm(x.view(x.shape[0], 1, -1), action.view(action.shape[0], -1, 1)).view(-1) \
                            for i, x in enumerate(gs)]
        losses = [torch.mean((c_next[:, i] - c_next_predicted[i]) ** 2) for i in range(self._num_constraints)]

        return losses

    def _update_batch(self, batch):
        batch = self._buf.sample_batch()

        # Update critic
        for_each(lambda x: x.zero_grad(), self._optimizers)
        losses = self._evaluate_batch(batch)
        for_each(lambda x: x.backward(), losses)
        for_each(lambda x: x.step(), self._optimizers)

        return np.asarray([x.item() for x in losses])

    def evaluate(self):
        # Sample steps
        self._sample_steps(self._config.evaluation_steps)

        self._eval_mode()
        # compute losses
        losses = [list(map(lambda x: x.item(), self._evaluate_batch(batch))) for batch in \
                self._replay_buffer.get_sequential(self._config.batch_size)]




        self._eval_global_step += 1

        self._train_mode()

        print(f"Validation completed, average loss {losses}")

    def get_safe_action(self, observation, action, c):
        # Find the values of G
        self._eval_mode()
        g = [x(self._as_tensor(observation).view(1, -1)) for x in self._models]
        self._train_mode()

        # Fidn the lagrange multipliers
        g = [x.data.numpy().reshape(-1) for x in g]
        multipliers = [(np.dot(g_i, action) + c_i) / np.dot(g_i, g_i) for g_i, c_i in zip(g, c)]
        multipliers = [np.clip(x, 0, np.inf) for x in multipliers]

        # Calculate correction
        correction = np.max(multipliers) * g[np.argmax(multipliers)]

        action_new = action - correction

        return action_new

    def train(self):

        start_time = time.time()

        print("==========================================================")
        print("Initializing constraint model training...")
        print("----------------------------------------------------------")
        print(f"Start time: {datetime.fromtimestamp(start_time)}")
        print("==========================================================")


        number_of_steps = self._config.steps_per_epoch * self._config.epochs

        for epoch in range(self._config.epochs):
            # Just sample episodes for the whole epoch
            self._sample_steps(self._config.steps_per_epoch)

            # Do the update from memory
            losses = np.mean(np.concatenate([self._update_batch(batch) for batch in \
                    self._replay_buffer.get_sequential(self._config.batch_size)]).reshape(-1, self._num_constraints), axis=0)



            (seq(self._models)
                    .zip_with_index() # (model, index)
                    .map(lambda x: (f"constraint_model_{x[1]}", x[0])) # (model_name, model)
                    .flat_map(lambda x: [(x[0], y) for y in x[1].named_parameters()]) # (model_name, (param_name, param_data))
                    .map(lambda x: (f"{x[0]}_{x[1][0]}", x[1][1])) # (modified_param_name, param_data)
             )

            self._train_global_step += 1

            print(f"Finished epoch {epoch} with losses: {losses}. Running validation ...")
            self.evaluate()
            print("----------------------------------------------------------")


        print("==========================================================")
        print(f"Finished training constraint model. Time spent: {(time.time() - start_time) // 1} secs")
        print("==========================================================")
