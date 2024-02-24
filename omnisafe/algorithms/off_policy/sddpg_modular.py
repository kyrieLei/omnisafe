# Copyright 2023 OmniSafe Team. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Implementation of the Deep Deterministic Policy Gradient algorithm."""

from __future__ import annotations

import time
from typing import Any

import torch
from torch import nn
from torch.nn.utils.clip_grad import clip_grad_norm_
import numpy as np
from omnisafe.adapter import OffPolicyAdapter
from omnisafe.algorithms import registry
from omnisafe.algorithms.base_algo import BaseAlgo
from omnisafe.algorithms.off_policy.sddpg import SDDPG
from omnisafe.common.buffer import VectorOffPolicyBuffer
from omnisafe.common.logger import Logger
from omnisafe.models.actor_critic.constraint_actor_q_critic import ConstraintActorQCritic
from omnisafe.models.actor_critic.actor_q_critic import ActorQCritic
from omnisafe.common.safety_projection import C_Critic





@registry.register
# pylint: disable-next=too-many-instance-attributes,too-few-public-methods
class SDDPG_Modular(SDDPG):


    def _update_reward_critic(
        self,
        obs: torch.Tensor,
        action: torch.Tensor,
        reward: torch.Tensor,
        done: torch.Tensor,
        next_obs: torch.Tensor,
    ) -> None:
        """Update reward critic.

        - Get the TD loss of reward critic.
        - Update critic network by loss.
        - Log useful information.

        Args:
            obs (torch.Tensor): The ``observation`` sampled from buffer.
            action (torch.Tensor): The ``action`` sampled from buffer.
            reward (torch.Tensor): The ``reward`` sampled from buffer.
            done (torch.Tensor): The ``terminated`` sampled from buffer.
            next_obs (torch.Tensor): The ``next observation`` sampled from buffer.
        """
        with torch.no_grad():
            next_action = self._actor_critic.actor.predict(next_obs, deterministic=True)
            next_q_value_r = self._actor_critic.target_reward_critic(next_obs, next_action)[0]
            target_q_value_r = reward + self._cfgs.algo_cfgs.gamma * (1 - done) * next_q_value_r
        q_value_r = self._actor_critic.reward_critic(obs, action)[0]
        loss = nn.functional.mse_loss(q_value_r, target_q_value_r)

        if self._cfgs.algo_cfgs.use_critic_norm:
            for param in self._actor_critic.reward_critic.parameters():
                loss += param.pow(2).sum() * self._cfgs.algo_cfgs.critic_norm_coeff
        self._logger.store(
            {
                'Loss/Loss_reward_critic': loss.mean().item(),
                'Value/reward_critic': q_value_r.mean().item(),
            },
        )
        self._actor_critic.reward_critic_optimizer.zero_grad()
        loss.backward()

        if self._cfgs.algo_cfgs.max_grad_norm:
            clip_grad_norm_(
                self._actor_critic.reward_critic.parameters(),
                self._cfgs.algo_cfgs.max_grad_norm,
            )
        self._actor_critic.reward_critic_optimizer.step()

    def _update_cost_critic(
        self,
        obs: torch.Tensor,
        action: torch.Tensor,
        cost: torch.Tensor,
        done: torch.Tensor,
        next_obs: torch.Tensor,
    ) -> None:
        """Update cost critic.

        - Get the TD loss of cost critic.
        - Update critic network by loss.
        - Log useful information.

        Args:
            obs (torch.Tensor): The ``observation`` sampled from buffer.
            action (torch.Tensor): The ``action`` sampled from buffer.
            cost (torch.Tensor): The ``cost`` sampled from buffer.
            done (torch.Tensor): The ``terminated`` sampled from buffer.
            next_obs (torch.Tensor): The ``next observation`` sampled from buffer.
        """
        with torch.no_grad():
            next_action = self._actor_critic.actor.predict(next_obs, deterministic=True)
            next_q_value_c = self._actor_critic.target_cost_critic(next_obs, next_action)[0]
            target_q_value_c = cost + self._cfgs.algo_cfgs.gamma * (1 - done) * next_q_value_c
        q_value_c = self._actor_critic.cost_critic(obs, action)[0]
        loss = nn.functional.mse_loss(q_value_c, target_q_value_c)

        if self._cfgs.algo_cfgs.use_critic_norm:
            for param in self._actor_critic.cost_critic.parameters():
                loss += param.pow(2).sum() * self._cfgs.algo_cfgs.critic_norm_coeff

        self._actor_critic.cost_critic_optimizer.zero_grad()
        loss.backward()

        if self._cfgs.algo_cfgs.max_grad_norm:
            clip_grad_norm_(
                self._actor_critic.cost_critic.parameters(),
                self._cfgs.algo_cfgs.max_grad_norm,
            )
        self._actor_critic.cost_critic_optimizer.step()

        self._logger.store(
            {
                'Loss/Loss_cost_critic': loss.mean().item(),
                'Value/cost_critic': q_value_c.mean().item(),
            },
        )


    def _update_actor(  # pylint: disable=too-many-arguments
        self,
        obs
    ):
        """Update actor.

        - Get the loss of actor.
        - Update actor by loss.
        - Log useful information.
        """

        loss = self._loss_pi(obs)
        self._actor_critic.actor_optimizer.zero_grad()
        loss.backward()
        if self._cfgs.algo_cfgs.max_grad_norm:
            clip_grad_norm_(
                self._actor_critic.actor.parameters(),
                self._cfgs.algo_cfgs.max_grad_norm,
            )
        self._actor_critic.actor_optimizer.step()
        self._logger.store(
            {
                'Loss/Loss_pi': loss.mean().item(),
            },
        )

    def auto_grad(self,objective,net,to_numpy=True):
        """
        Get the gradient of the objective with respect to the parameters of the network
        """
        grad = torch.autograd.grad(objective,net)
        if to_numpy:
            return torch.cat([val.flatten() for val in grad], axis=0).detach().cpu().numpy()
        else:
            return torch.cat([val.flatten() for val in grad], axis=0)

    def auto_hession_x(self,objective, net, x):

        jacob = self.auto_grad(objective,net,to_numpy=False)

        return self.auto_grad(torch.dot(jacob, x), to_numpy=True)
    def conjugate_gradient(self,Ax, b, cg_iters=100):
        EPS=1e-8
        x = np.zeros_like(b)
        r = b.copy()  # Note: should be 'b - Ax', but for x=0, Ax=0. Change if doing warm start.
        p = r.copy()
        r_dot_old = np.dot(r, r)
        for _ in range(cg_iters):
            z = Ax(p)
            alpha = r_dot_old / (np.dot(p, z) + EPS)
            x += alpha * p
            r -= alpha * z
            r_dot_new = np.dot(r, r)
            p = r + (r_dot_new / r_dot_old) * p
            r_dot_old = r_dot_new
            # early stopping
            if np.linalg.norm(p) < EPS:
                break
        return x


    def _loss_pi(
        self,
        obs: torch.Tensor,
    ) -> torch.Tensor:
        d0=100
        EPS=1e-8

        action = self._actor_critic.actor.predict(obs, deterministic=True)
        Q_d = self._actor_critic.cost_critic(obs, action)[0]
        Q = self._actor_critic.reward_critic(obs,action)[0]
        grad_Q_d=C_Critic.discount_cumsum(self.auto_grad(self._actor_critic.actor.log_prob(action),Q_d*self._actor_critic.actor.log_prob(action)),self.self._cfgs.algo_cfgs.gamma)
        grad_Q=C_Critic.discount_cumsum(self.auto_grad(self._actor_critic.actor.log_prob(action),Q*self._actor_critic.actor.log_prob(action)),self.self._cfgs.algo_cfgs.gamma)

        distribution = self._actor_critic.actor(obs)
        epislon = (1 - self._cfgs.algo_cfgs.gamma)(d0 - Q_d)
        beta=self._cfgs.algo_cfgs.entropy_coef
        entropy = distribution.entropy().mean().item()

        # SafeLayer policy update core impelmentation

        Hx = lambda x: self.auto_hession_x(entropy, self._actor_critic.actor.parameters(), torch.FloatTensor(x))
        x_hat = self.conjugate_gradient(Hx, grad_Q_d)

        s = grad_Q_d.T @ x_hat
        lambda_star = (-beta*epislon-grad_Q.T@x_hat)/(s+EPS)
        pi = torch.exp(self._actor_critic.actor.log_prob(action))
        inner = self.auto_grad(Q,action)+lambda_star*self.auto_grad(Q_d,action)
        loss=(pi*inner).mean().item()


        return loss

    def _log_when_not_update(self) -> None:
        """Log default value when not update."""
        self._logger.store(
            {
                'Loss/Loss_reward_critic': 0.0,
                'Loss/Loss_pi': 0.0,
                'Value/reward_critic': 0.0,
            },
        )
        if self._cfgs.algo_cfgs.use_cost:
            self._logger.store(
                {
                    'Loss/Loss_cost_critic': 0.0,
                    'Value/cost_critic': 0.0,
                },
            )
