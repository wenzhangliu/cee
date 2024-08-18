# from stable_baselines3.common import logger
# from stable_baselines3.common.logger import Figure
import torch
from torch.distributions.categorical import Categorical
import itertools
import numpy as np
import matplotlib.pyplot as plt


def plot_confusion_matrix(cmtx, num_classes, class_names=None, figsize=None):
    """
    A function to create a colored and labeled confusion matrix matplotlib figure
    given true labels and preds.
    Args:
        cmtx (ndarray): confusion matrix.
        num_classes (int): total number of classes.
        class_names (Optional[list of strs]): a list of class names.
        figsize (Optional[float, float]): the figure size of the confusion matrix.
            If None, default to [6.4, 4.8].

    Returns:
        img (figure): matplotlib figure.
    """
    if class_names is None or type(class_names) != list:
        class_names = [str(i) for i in range(num_classes)]

    # if figsize[0] > 640:
    #     figsize =[640, 480]
    figure = plt.figure(figsize=figsize)
    plt.imshow(cmtx, interpolation="nearest", cmap=plt.cm.Blues)
    plt.title("Action Mask Factor matrix")
    plt.colorbar()
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45)
    plt.yticks(tick_marks, class_names)

    # Use white text if squares are dark; otherwise black.
    threshold = cmtx.max() / 2.0
    for i, j in itertools.product(range(cmtx.shape[0]), range(cmtx.shape[1])):
        color = "white" if cmtx[i, j] > threshold else "black"
        plt.text(
            j,
            i,
            format(cmtx[i, j], ".2f") if cmtx[i, j] != 0 else ".",
            horizontalalignment="center",
            color=color,
        )

    plt.tight_layout()
    plt.ylabel("Action a_t")
    plt.xlabel("Action b_t")

    return figure

class MfModelTrainer:
    """
    A custom callback that derives from ``BaseCallback``.

    :param verbose: (int) Verbosity level 0: not output 1: info 2: debug
    """
    def __init__(self, mf_model, discrete, new_logger,cat_dim=1):
        self.mf_model = mf_model
        self.cat_dim = cat_dim
        self.discrete = discrete
        self.new_logger = new_logger
        self.nupdates = 0

    def train_step(self, batch, **kwargs):
        self.nupdates += 1
        if self.discrete:
            self.train_step_discrete(batch, **kwargs)
        else:
            raise ValueError

    # self.mf_trainer.train_step(replay_data, pi=act_pi, action_module=self.action_trainer.action_model.q_net,
    #                            max_grad_norm=None)
    def train_step_discrete(self, batch, pi, action_module, max_grad_norm):
        divergence_list = ["KL","TV","Hellinger"]
        divergence_mode = divergence_list[0]
        # print(divergence_mode)
        # Input :
        # batch:
        # pi:pi(s)
        # action_module: p(a|s,s',pi(s))
        # for example:
        # action_model_logits = action_module(new_x)
        # action_model_probs = th.nn.Softmax(dim=1)(action_model_logits)

        # Output:
        # M(s, a, b)

        # predicted:
        # M(s, a) = mf (action_dim)


        # pipeline:
        # 1.calculate p(b|s,s',pi(s)) for every action b
        # 2.calculate mf_b = log[p(b|s,s',pi(s))] - log[pi(b|s)] for every b
        # 3.Back Propagation: L2 loss :||M(s,a) - mf ||

        # 1.calculate p(b|s,s',pi(s)) for every action b
        x = torch.cat((batch.observations, batch.next_observations), dim=self.cat_dim).float()
        # we use the "q_net" output to get action probabilities
        new_x = {"image": x, "vector": pi.float()}
        action_model_logits = action_module(new_x)
        action_model_probs = torch.nn.Softmax(dim=1)(action_model_logits)
        epsilon = 1e-8
        # predicted:
        # M(s, a) = mf (action_dim)
        real_action = torch.nn.functional.one_hot(batch.actions, pi.shape[1]).squeeze(dim=1)
        sa = {"image": batch.observations.float(), "vector": real_action.float()}
        mf_predicted = self.mf_model.q_net(sa)

        if divergence_mode == "KL":
            # 2.calculate mf_b = log[p(b|s,s',pi(s))] - log[pi(b|s)] for every b
            action_model_log_probs = torch.log(action_model_probs + epsilon)
            log_pi = torch.log(pi + epsilon)
            with torch.no_grad():
                mf_target = action_model_log_probs - log_pi


        elif divergence_mode == "TV":
            # 2.calculate vector_b = p(b|s,s',pi(s))/ pi(b|s)  for every b
            vector_b = (action_model_probs + epsilon) / (pi + epsilon)

            # 3.calculate constant_a = p(a|s,s',pi(s))/ pi(a|s)
            index = batch.actions
            action_model_probs_a = action_model_probs.gather(1,index)
            pi_a = pi.gather(1,index)
            constant_a = (action_model_probs_a + epsilon) / (pi_a + epsilon)

            # 4.calculate mf_b = |[p(b|s,s',pi(s)) * pi(a|s)] / [pi(b|s) * p(a|s,s',pi(s))]  - 1|
            with torch.no_grad():
                mf_target = torch.abs(vector_b / constant_a - 1)


        elif divergence_mode == "Hellinger":
            # 2.calculate vector_b = p(b|s,s',pi(s))/ pi(b|s)  for every b
            vector_b = (action_model_probs + epsilon) / (pi + epsilon)

            # 3.calculate constant_a = p(a|s,s',pi(s))/ pi(a|s)
            index = batch.actions
            action_model_probs_a = action_model_probs.gather(1, index)
            pi_a = pi.gather(1, index)
            constant_a = (action_model_probs_a + epsilon) / (pi_a + epsilon)

            # 4.calculate mf_b = sqrt( [p(b|s,s',pi(s)) * pi(a|s)] / [pi(b|s) * p(a|s,s',pi(s))])
            with torch.no_grad():
                mf_target = torch.sqrt( vector_b / constant_a )

        # 3.Back Propagation: L2 loss :||M(s,a) - mf ||
        loss_func = torch.nn.MSELoss()
        loss = loss_func(mf_predicted, mf_target)
        if loss < 0:
            print(loss)

        # 4.symmetry loss: L2 loss : ||M(s,a_i)[j] - M(s,a_j)[i]||
        batch_size = len(batch.actions)
        beta= 1
        if divergence_mode == "TV" or "Hellinger":
            i_index = torch.randint(low=0, high=pi.shape[1], size=[batch_size]).to("cuda:"+str(batch.actions.get_device()))
            j_index = torch.randint(low=0, high=pi.shape[1], size=[batch_size]).to("cuda:"+str(batch.actions.get_device()))
            i_index_one_hot = torch.nn.functional.one_hot(i_index, pi.shape[1])
            j_index_one_hot = torch.nn.functional.one_hot(j_index, pi.shape[1])
            sa_i = {"image": batch.observations.float(), "vector": i_index_one_hot.float()}
            mf_predicted_sai = self.mf_model.q_net(sa_i)
            sa_j = {"image": batch.observations.float(), "vector": j_index_one_hot.float()}
            mf_predicted_saj = self.mf_model.q_net(sa_j)

            # M(s,a_i)[j]
            M_i_j = mf_predicted_sai.gather(1, j_index.unsqueeze(1))
            # M(s,a_j)[i]
            M_j_i = mf_predicted_saj.gather(1, i_index.unsqueeze(1))

            s_loss = loss_func(M_i_j, M_j_i)

        if divergence_mode == "KL":
            s_loss = 0
        total_loss = loss + beta * s_loss
        # Optimize the action model
        self.mf_model.optimizer.zero_grad()
        total_loss.backward()
        # Clip gradient norm
        if max_grad_norm:
            torch.nn.utils.clip_grad_norm_(self.mf_model.parameters(), max_grad_norm)
        self.mf_model.optimizer.step()
        logger = self.new_logger
        logger.record("mf model/mf_loss", loss.item())

        if divergence_mode != "KL":
            logger.record("mf model/symmetric_loss", s_loss.item())
        logger.record("mf model/total_loss", total_loss.item())
        logger.record("mf model/n_updates", self.nupdates)

        # # 4.calculate the mean mf factor in batch of every action:
        # log_interval = 2000
        # n_actions = pi.shape[1]
        # temp = np.ceil(n_actions/8)
        # figsize = [temp * 6.4, temp * 4.8]
        # with torch.no_grad():
        #     eps = 1e-4
        #     # M(at,bt) = log(p(bt|st,f(st,at))
        #     Mask_factor = torch.log(action_model_probs + eps) - torch.log(pi + eps)
        #     # for different state s,if mask is the same, we can calculate the mask
        #     mf_list = []
        #     temp = [-10 for i in range(n_actions)]
        #     for aa in range(n_actions):
        #         mf_aa = Mask_factor[torch.squeeze(batch.actions == aa, dim=1)]
        #         if len(mf_aa) > 0:
        #             mean_mf_aa = torch.mean(mf_aa, dim=0)
        #             mf_list.append(mean_mf_aa.tolist())
        #         else:
        #             mf_list.append(temp)
        #     mf_fig = plot_confusion_matrix(np.array(mf_list), pi.shape[1],figsize=figsize)
        #     logger.record("mf target/mf_table", Figure(mf_fig, close=True), exclude=("stdout", "log", "json", "csv"))
        #     plt.close()
        #
        # with torch.no_grad():
        #     mf_predict_list = []
        #     for aa in range(n_actions):
        #         # all action is aa
        #         test_actions = torch.ones_like(batch.actions) * aa
        #         test_actions_onehot = torch.nn.functional.one_hot(test_actions, pi.shape[1]).squeeze(dim=1)
        #         sa = {"image": batch.observations.float(), "vector": test_actions_onehot.float()}
        #         mf_predicted_aa = self.mf_model.q_net(sa)
        #         mean_mf_predicted_aa = torch.mean(mf_predicted_aa,dim=0)
        #         mf_predict_list.append(mean_mf_predicted_aa.tolist())
        #     mf_fig = plot_confusion_matrix(np.array(mf_predict_list), pi.shape[1],figsize=figsize)
        #     logger.record("mf predict/mf_table", Figure(mf_fig, close=True), exclude=("stdout", "log", "json", "csv"))
        #     plt.close()