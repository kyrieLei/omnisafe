
from __future__ import annotations

import torch
from omnisafe.algorithms import registry
from omnisafe.algorithms.off_policy.sddpg import SDDPG
from torch.nn.utils.clip_grad import clip_grad_norm_
from omnisafe.utils.config import ModelConfig
from omnisafe.common.Optimizer import New_optim
from omnisafe.common.buffer import VectorOffPolicyBuffer


@registry.register
class SDDPG_Modular(SDDPG):

    def _update_actor(
        self,
        obs
    ):
        loss, inner= self._loss_pi(obs)
        self._actor_critic.actor_optimizer.zero_grad()
        loss.backward()
        if self._cfgs.algo_cfgs.max_grad_norm:
            clip_grad_norm_(
                self._actor_critic.actor.parameters(),
                self._cfgs.algo_cfgs.max_grad_norm,
            )
        self._actor_critic.actor_optimizer.step(inner=inner)
        self._logger.store(
            {
                'Loss/Loss_pi': loss.mean().item(),
            },
        )

    def _loss_pi(
        self,
        obs: torch.Tensor,
    ) -> torch.Tensor:


        action = self._actor_critic.actor.predict(obs, deterministic=True)

        Q = self._actor_critic.reward_critic(obs,action)[0]

        pi = torch.exp(self._actor_critic.actor.log_prob(action))
        inner = self.auto_grad(Q,self._actor_critic.actor.parameters(), retain_graph=True)
        loss=pi

        return loss,inner


