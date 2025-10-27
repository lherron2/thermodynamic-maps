import torch
import os
import numpy as np
from torch.func import vmap, jvp

import logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

def temperature_density_rescaling(std_temp, ref_temp):
    """
    Calculate temperature density rescaling factor.

    Args:
        std_temp (float): The standard temperature.
        ref_temp (float): The reference temperature.

    Returns:
        float: The temperature density rescaling factor.
    """
    return (std_temp / ref_temp).pow(0.5)


def identity(t, *args, **kwargs):
    """
    Identity function.

    Args:
        t: Input tensor.

    Returns:
        t: Input tensor.
    """
    return t


RESCALE_FUNCS = {
    "density": temperature_density_rescaling,
    "no_rescale": identity,
}


class DiffusionModel:
    """
    Base class for diffusion models.
    """

    def __init__(
        self,
        diffusion_process,
        backbone,
        loader,
        prior,
        network_pred_type,
        rescale_func_name="no_rescale",
        RESCALE_FUNCS=RESCALE_FUNCS,
        **kwargs
    ):
        """
        Initialize a DiffusionModel.

        Args:
            diffusion_process: The diffusion process.
            backbone: The backbone model.
            loader: Data loader.
            pred_type: Type of prediction.
            prior: Prior distribution.
            control_ref (float): Control reference temperature.
            rescale_func_name (str): Name of the rescaling function.
            RESCALE_FUNCS (dict): Dictionary of rescaling functions.
        """
        self.loader = loader
        self.BB = backbone
        self.DP = diffusion_process
        self.pred_type = network_pred_type
        self.rescale_func = RESCALE_FUNCS[rescale_func_name]
        self.prior = prior

    def noise_batch(self, b_t, t, prior, **prior_kwargs):
        """
        Wrapper which calls applies the (marginal) transition kernel
        of the forward noising process.
        """
        x, noise, score = self.DP.kernel('jump', b_t, t, prior, likelihood=False, **prior_kwargs)
        return x, noise

    def denoise_batch(self, b_t, t):
        """
        Wrapper which calls applies the (marginal) transition kernel
        of the reverse noising process.
        """
        x, noise, score = self.DP.kernel('reverse_jump', b_t, t, self.BB, self.pred_type)
        return x, noise

    def sample_times(self, num_times):
        """
        Randomly sample times from the time-discretization of the
        diffusion process.
        """
        return torch.randint(
            low=0, high=self.DP.num_diffusion_timesteps-1, size=(num_times,)
        ).long()

    @staticmethod
    def get_adjacent_times(times, reverse=False):
        """
        Pairs t with t_next for all times in the time-discretization
        of the diffusion process.

        Args:
            times (torch.Tensor): Array of time steps.
            reverse (bool): Direction of pairing.

        Returns:
            list: List of (t, t_next) tuples.
        """
        if reverse:
            times_next = torch.cat((torch.tensor([0], dtype=torch.long), times[:-1]))
            return list(zip(reversed(times), reversed(times_next)))
        else:
            return list(zip(times[:-1], times[1:]))


class DiffusionTrainer(DiffusionModel):
    """
    Subclass of a DiffusionModel: A trainer defines a loss function and
    performs backprop + optimizes model outputs.
    """

    def __init__(
        self,
        diffusion_process,
        backbone,
        train_loader,
        prior,
        network_pred_type='noise',
        model_dir=None,
        test_loader = None,
        optim=None,
        scheduler=None,
        rescale_func_name="density",
        RESCALE_FUNCS=RESCALE_FUNCS,
        device=0,
        identifier="model",
        train_sampler=None,
        test_sampler=None,
    ):
        """
        Initialize a DiffusionTrainer.

        Args:
            diffusion_process: The diffusion process.
            backbone: The backbone model.
            loader: Data loader.
            pred_type: Type of prediction.
            prior: Prior distribution.
            optim: Optimizer.
            scheduler: Learning rate scheduler.
            rescale_func_name (str): Name of the rescaling function.
            RESCALE_FUNCS (dict): Dictionary of rescaling functions.
        """
        super().__init__(
            diffusion_process,
            backbone,
            train_loader,
            prior,
            network_pred_type,
            rescale_func_name,
            RESCALE_FUNCS,
        )

        self.model_dir = model_dir
        if self.model_dir:
            os.makedirs(self.model_dir, exist_ok=True)
        self.identifier = identifier
        self.test_loader = test_loader
        self.train_loader = train_loader
        self.train_losses = []
        self.test_losses = []
        self.train_sampler = train_sampler
        self.test_sampler = test_sampler

        if self.test_loader is not None: logger.info("From DiffusionTrainer: Test loader found")
        else: logger.info("From DiffusionTrainer: No test loader. Ignoring")

    def loss_function(self, e, e_pred, loss_type="l2", weights=None):
        """
        Loss function can be the l1-norm, l2-norm, or the VLB (weighted l2-norm).

        Args:
            e: Actual data.
            e_pred: Predicted data.
            weight: Weight factor.
            loss_type (str): Type of loss function.

        Returns:
            float: The loss value.
        """
        def l1_loss(e, e_pred, weights):
            return (e - e_pred).abs().sum(sum_indices)

        def l2_loss(e, e_pred, weights):
            axes=tuple(list(np.arange(len(e.shape)-1)+1))
            l2_norm = (e - e_pred).pow(2).sum(axes).pow(0.5)
            weighted_loss = (l2_norm*weights).mean()
            return weighted_loss
            # else:
                # return (e - e_pred).pow(2).sum(axes).pow(0.5).mean()
                
                

        def VLB_loss(e, e_pred, weights):
            return (weight * ((e - e_pred).pow(2).sum(sum_indices)).pow(0.5)).mean()

        def smooth_l1_loss(e, e_pred, weights):
            return torch.nn.functional.smooth_l1_loss(e, e_pred)

        loss_dict = {"l1": l1_loss, "l2": l2_loss, "VLB": VLB_loss, "smooth_l1": smooth_l1_loss}

        return loss_dict[loss_type](e, e_pred, weights)

    def train(
        self,
        num_epochs,
        grad_accumulation_steps=1,
        print_freq=None,
        batch_size=128,
        loss_type="l2",
    ):
        """
        Trains a diffusion model.

        Args:
            num_epochs (int): Number of training epochs.
            grad_accumulation_steps (int): Number of gradient accumulation steps.
            print_freq (int): Frequency of printing training progress.
            batch_size (int): Batch size.
            loss_type (str): Type of loss function.
        """

        train_loader = torch.utils.data.DataLoader(
            self.train_loader,
            batch_size=batch_size,
            sampler=self.train_sampler,
            shuffle=False,
            pin_memory=True
        )
        if self.test_loader:
            test_loader = torch.utils.data.DataLoader(
                self.test_loader,
                batch_size=batch_size,
                sampler=self.test_sampler,
                shuffle=False,
                pin_memory=True
            )

        for epoch in range(num_epochs):
            epoch_train_loss = []
            epoch += self.BB.start_epoch
            for i, (index, temperatures, b) in enumerate(train_loader, 0):
                if b.size(0) != batch_size:
                    continue
                t = self.sample_times(b.size(0))
                t_prev = t - 1
                t_prev[t_prev == -1] = 0
                target, output = self.train_step(b, t, self.prior, 
                    batch_size=len(b), temperatures=temperatures, sample_type="from_data") # prior kwargs
                energy_no_offset = temperatures[:,0,0].squeeze()
                weights = torch.ones_like(energy_no_offset)
                loss = (self.loss_function(target, output, loss_type=loss_type, weights=weights) / grad_accumulation_steps)

                if i % grad_accumulation_steps == 0:
                    self.BB.optim.zero_grad()
                    epoch_train_loss.append(loss.detach().cpu().numpy())
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.BB.model.parameters(), 1.)
                    self.BB.optim.step()

                if print_freq:
                    if i % print_freq == 0:
                        print(f"step: {i}, loss {loss.detach():.3f}")

            if self.test_loader:
                with torch.no_grad():
                    epoch_test_loss = []
                    for i, (index, temperatures, b) in enumerate(test_loader, 0):
                        t = self.sample_times(b.size(0))
                        t_prev = t - 1
                        t_prev[t_prev == -1] = 0
                        target, output = self.train_step(b, t, self.prior, 
                            batch_size=len(b), temperatures=temperatures, sample_type="from_data")
                        loss = self.loss_function(target, output, loss_type=loss_type)
                        epoch_test_loss.append(loss.detach().cpu().numpy())

            self.train_losses.append(np.mean(epoch_train_loss))
            if self.test_loader:
                self.test_losses.append(np.mean(epoch_test_loss))
                print(f"epoch: {epoch} | train loss: {self.train_losses[-1]:.3f} | test loss: {self.test_losses[-1]:.3f}")
            else:
                print(f"epoch: {epoch} | train loss: {self.train_losses[-1]:.3f}")

            if self.model_dir:
                self.BB.save_state(self.model_dir, 'latest', identifier=self.identifier)

    def train_step(self, b, t, prior, **kwargs):
        """
        Training step.

        Args:
            b: Input batch.
            t: Sampled times.
            prior: Prior distribution.
            kwargs: Additional keyword arguments.

        Returns:
            Tuple: (noise, noise_pred)
        """
        b_t, noise = self.noise_batch(b, t, prior, **kwargs)
        b_0, noise_pred = self.denoise_batch(b_t, t)
        if self.pred_type == "noise":
            return noise, noise_pred
        elif self.pred_type == "x0":
            return b, b_0


class DiffusionSampler(DiffusionModel):
    """
    Subclass of a DiffusionModel: A sampler generates samples from random noise.
    """

    def __init__(
        self,
        diffusion_process,
        backbone,
        loader,
        prior,
        network_pred_type='noise',
        sample_dir=None,
        rescale_func_name="density",
        RESCALE_FUNCS=RESCALE_FUNCS,
        **kwargs
    ):
        """
        Initialize a DiffusionSampler.

        Args:
            diffusion_process: The diffusion process.
            backbone: The backbone model.
            loader: Data loader.
            pred_type: Type of prediction.
            prior: Prior distribution.
            rescale_func_name (str): Name of the rescaling function.
            RESCALE_FUNCS (dict): Dictionary of rescaling functions.
        """
        super().__init__(
            diffusion_process,
            backbone,
            loader,
            prior,
            network_pred_type,
            rescale_func_name,
            RESCALE_FUNCS,
            **kwargs
        )
        self.sample_dir = sample_dir
        if self.sample_dir:
            os.makedirs(self.sample_dir, exist_ok=True)

    def build_channel_dict(self, batch_size, prior, temperature):
        """
        Build a dictionary of the conditional values for each channel.

        Args:
            batch_size: The size of the batch.
            prior: The prior object.
            temperature: The temperature, can be a scalar or a vector.

        Returns:
            Dict: Dictionary of the conditional values for each channel.
        """
        fluct_channels = prior.channels_info["fluctuation"]
        num_fluct_channels = len(fluct_channels)
        channel_slice = [batch_size] + [1] + list(prior.shape[1:])  # each channel is treated individually
        channel_dict = {}
        temperatures = torch.full((num_fluct_channels,), temperature)

        for channel, temp in zip(fluct_channels, temperatures):
            rmsf = prior.multilinear_fit.evaluate(np.array([temperature])).squeeze()
            channel_dict[channel] = rmsf
        return channel_dict

    def _xt_and_divergence(self, xt, step_fn, n_vecs=4):
        # Draw all probe vectors at once  (n_vecs, batch, …)
        v = torch.randn((n_vecs, *xt.shape), device=xt.device)
    
        # vmap runs *one* forward of f and re-uses its intermediates
        (xt_next, Jv) = vmap(
            lambda v_i: jvp(step_fn, (xt,), (v_i,)),
            randomness='same'
        )(v)                                  # shapes: (n_vecs, batch, …)
    
        # xt_next is repeated n_vecs times; pick the first
        xt_next = xt_next[0]
    
        div = (Jv * v).flatten(2).sum(2).mean(0)   # avg over probes → (batch,)
        return xt_next.detach(), div.detach()


    def sample_batch(
        self,
        x0='prior',
        mode='reverse',
        likelihood=False,
        control_dict=None,
        gamma=None,
        network_pred_type='noise',
        n_noise_vecs=10,
        **prior_kwargs
    ):
        """
        Sample a batch of data in forward or reverse mode.
    
        Args:
            x0 (torch.Tensor, optional): Initial sample for forward mode.
            mode (str): 'forward' or 'reverse' sampling.
            likelihood (bool): Whether to compute likelihood-related quantities.
            control_dict (dict, optional): Control parameters for the sampling process.
            gamma (float, optional): Control parameter for sampling.
            pred_type (str): Type of prediction ('noise' by default).
            n_noise_vecs (int): Number of noise vectors for divergence computation.
            **prior_kwargs: Additional keyword arguments for the prior and sampling.
    
        Returns:
            tuple: 
                - torch.Tensor: Sampled batch.
                - list: List of divergence values (if likelihood is True).
        """
        if mode not in {'forward', 'reverse'}:
            raise ValueError("mode must be 'forward' or 'reverse'")
    
        # Initialize the sample based on the mode
        
        xt, div_list = self._initialize_sample(x0, mode, likelihood, prior_kwargs)

        if mode == 'forward':
            output_dict = {'target': xt.detach().numpy()}
        elif mode == 'reverse':
            output_dict = {'prior': xt.detach().numpy()}
    
        # Retrieve time pairs based on the mode
        reverse_flag = (mode == 'reverse')
        time_pairs = self.get_adjacent_times(self.DP.times, reverse=reverse_flag)

        temps = prior_kwargs.get("temperatures")
        if torch.is_tensor(temps):
            prior_kwargs["temperatures"] = temps.detach().cpu().numpy()
    
        # Iterate through each time step and perform sampling
        for current_time, next_time in time_pairs:
            # Create tensors for current and next time steps
            t_tensor, t_next_tensor = self._create_time_tensors(
                current_time, next_time, prior_kwargs['batch_size']
            )

            def step_fn(x):
                return self.DP.step(
                    x=x,
                    t=t_tensor,
                    t_next=t_next_tensor,
                    prior=self.prior,
                    mode=mode,
                    control_dict=control_dict,
                    gamma=gamma,
                    network_pred_type=network_pred_type,
                    backbone=self.BB,
                    likelihood=likelihood,
                    **prior_kwargs
                )

            xt_next, div = self._xt_and_divergence(xt, step_fn)

            # If likelihood is True, compute and store divergence
            if likelihood:
                # div = self._compute_divergence(xt, xt_next, n_noise_vecs)
                div_list.append(div)
                xt = xt_next.detach().requires_grad_(True)
            else:
                xt = xt_next.detach()
            # print(f'Completed time step {current_time}')
            
        if mode == "forward":
            # store the final latent sample without leaving the current device / dtype
            output_dict["prior"] = xt.detach().numpy()                            # (B, C, ...)
        
            if likelihood:
                # 1)  Δlog p – q  ← integrate divergence over time   (shape: (B,))
                div_t   = torch.stack(div_list, dim=0)                    # (T-1, B)
                betas_t = self.DP.betas[1:].to(div_t)                     # (T-1,)
                delta_log_pq = torch.trapz(div_t, betas_t, dim=0)         # (B,)
                output_dict["delta_log_pq"] = delta_log_pq
        
                # 2)  log q(z_T)  ← prior likelihood per sample (shape: (B,))
                log_q = self.prior.log_likelihood(
                    output_dict["prior"],
                    prior_kwargs["temperatures"],
                )                                                         # already (B,)
                output_dict["log_q"] = log_q
        
                # 3)  final likelihood under the score model
                output_dict["log_p"] = log_q + delta_log_pq               # (B,)

                
        elif mode == 'reverse':
            output_dict['target'] = xt.detach().numpy()
            if likelihood:
                div_t   = torch.stack(div_list, dim=0)                    # (T-1, B)
                betas_t = self.DP.betas[:].to(div_t)                     # (T-1,)
                # Integrate the divergence
                delta_log_pq = torch.trapz(div_t, betas_t.flip(0), dim=0)

                output_dict['delta_log_pq'] = delta_log_pq
                log_q = self.prior.log_likelihood(output_dict['prior'], 
                                                  prior_kwargs['temperatures'],
                                                  # sample_type='sample_from_fit'
                                                 )
                    # .mean(dim=(1, 2))
                output_dict['log_q'] = log_q
                output_dict['log_p'] = log_q + delta_log_pq
    
        return output_dict


    def _initialize_sample(self, x0, mode, likelihood, prior_kwargs):
        """
        Initialize the sample tensor based on the mode and provided initial sample.
    
        Args:
            x0 (torch.Tensor or None): Initial sample.
            mode (str): Sampling mode ('forward' or 'reverse').
            likelihood (bool): Whether to compute likelihood-related quantities.
            prior_kwargs (dict): Additional keyword arguments for the prior.
    
        Returns:
            tuple:
                - torch.Tensor: Initialized sample tensor.
                - list: Empty list for storing divergence values.
        """
        div_list = []
    
        if mode == 'reverse':
            if x0 == 'prior':
                xt = self.prior.sample(**prior_kwargs)
            else:
                xt = x0
        elif mode == 'forward':
            if x0 == 'prior':
                raise ValueError("Initial sample x0 must be provided in forward mode")
            xt = x0
        if likelihood:
            xt = xt.clone().detach().requires_grad_(True)
    
        return xt, div_list
    
    
    def _create_time_tensors(self, current_time, next_time, batch_size):
        """
        Create tensors filled with the current and next time steps.
    
        Args:
            current_time (int): Current time step.
            next_time (int): Next time step.
            batch_size (int): Size of the batch.
    
        Returns:
            tuple:
                - torch.Tensor: Tensor filled with current_time.
                - torch.Tensor: Tensor filled with next_time.
        """
        t = torch.full((batch_size,), current_time, dtype=torch.long)
        t_next = torch.full((batch_size,), next_time, dtype=torch.long)
        return t, t_next
    
    def save_batch(self, batch, save_prefix, temperature, save_idx):
        """
        Save a batch of samples.

        Args:
            batch: Batch of samples.
            save_prefix (str): Prefix for saving.
            temperature: Temperature for saving.
            save_idx (int): Index for saving.
        """
        save_path = os.path.join(self.sample_dir, f"{temperature}K")
        os.makedirs(save_path, exist_ok=True)
        np.savez_compressed(
            os.path.join(save_path, f"{save_prefix}_idx={save_idx}.npz"), traj=batch
        )
