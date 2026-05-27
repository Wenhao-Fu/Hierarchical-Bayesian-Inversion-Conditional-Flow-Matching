import copy
from datetime import datetime
import os
import numpy as np
import torch
from tqdm import tqdm
from utils import l2_loss, get_train_data, setup_logging
from modules_cond import UNet_conditional
import logging
from torch.utils.tensorboard import SummaryWriter
from torch import optim
from plot_field import plot_field

# flow_matching
from flow_matching.path.scheduler import CondOTScheduler
from flow_matching.path import AffineProbPath
from flow_matching.solver import Solver, ODESolver
from flow_matching.utils import ModelWrapper

logging.basicConfig(format="%(asctime)s - %(levelname)s: %(message)s", level=logging.INFO, datefmt="%I:%M:%S")


class FLOW_MATCHING:
    def __init__(self, image_size=64, device="cuda"):

        self.image_size = image_size
        self.device = device

    def sampling(self, model, num_samples, cond=None, step_size=0.05,
                 guidance_scale=1.0, return_intermediates=False):
        """
        Conditional sampling.

        Args:
            model: velocity model (nn.Module) or ModelWrapper. Should accept forward(x, t, **extras).
                   extras may include 'cond' (e.g. conditioning tensor / embedding).
            num_samples: int, batch size to sample.
            cond: optional conditioning tensor. Expected batch-first: (num_samples, ...).
                  If cond is provided and has batch size 1, it will be broadcasted to num_samples.
            step_size: ODE fixed step size (or None for adaptive solvers).
            guidance_scale: float, 1.0 means no CFG; >1.0 applies classifier-free guidance.
            return_intermediates: if True returns full trajectory per time_grid (like ODESolver).
        Returns:
            final sample (or sequence if return_intermediates=True)
        """

        logging.info(f"Sampling {num_samples} new images... (cond={'yes' if cond is not None else 'no'}, guidance={guidance_scale})")
        model.eval()

        class WrappedModel(ModelWrapper):
            def forward(self, x: torch.Tensor, t: torch.Tensor, **extras):
                batch_size = x.shape[0]
                if t.dim() == 0:
                    t = t.unsqueeze(0).expand(batch_size)
                image = self.model(x, t, **extras)
                return image

        wrapped_model = WrappedModel(model)

        # Build guided velocity callable for ODESolver.
        # ODESolver will call velocity_model(x, t, **extras)
        def guided_velocity(x, t, **extras):
            # extras can contain e.g. 'cond'
            # call wrapped model to obtain velocity under condition
            # mw returns a tensor shaped like x
            if guidance_scale == 1.0:
                return wrapped_model(x, t, **extras)
            else:
                # classifier-free guidance: combine cond & uncond predictions
                # unconditioned prediction: pass cond=None
                # note: ensure no side effects in mw when cond=None
                v_cond = wrapped_model(x, t, **extras)  # uses extras['cond'] if present
                # prepare extras for unconditioned call: drop cond key (or pass cond=None)
                extras_uncond = dict(extras)
                if 'conds' in extras_uncond:
                    extras_uncond['conds'] = None
                v_uncond = wrapped_model(x, t, **extras_uncond)
                return v_uncond + guidance_scale * (v_cond - v_uncond)

        # Time grid (you used 10 steps previously) — keep the same default / user-specified grid if desired
        T = torch.linspace(0, 1, 10)
        T = T.to(device=self.device)

        with torch.no_grad():

            x_init = torch.randn(num_samples, 1, self.image_size, self.image_size,
                                 device=self.device)
            # Create solver with our guided velocity callable
            solver = ODESolver(velocity_model=guided_velocity)
            # model_extras passed to velocity model at each call; include cond if provided
            model_extras = {}
            if cond is not None:
                model_extras['conds'] = cond.to(self.device)

            sol = solver.sample(time_grid=T, x_init=x_init, method='midpoint',
                                step_size=step_size,
                                return_intermediates=return_intermediates,
                                **model_extras)

            model.train()
            return sol

    def sampling_hm(self, model, num_samples, x_input, cond=None, step_size=0.05,
                    guidance_scale=1.0, return_intermediates=False):
        """
        Conditional sampling.

        Args:
            model: velocity model (nn.Module) or ModelWrapper. Should accept forward(x, t, **extras).
                   extras may include 'cond' (e.g. conditioning tensor / embedding).
            num_samples: int, batch size to sample.
            cond: optional conditioning tensor. Expected batch-first: (num_samples, ...).
                  If cond is provided and has batch size 1, it will be broadcasted to num_samples.
            step_size: ODE fixed step size (or None for adaptive solvers).
            guidance_scale: float, 1.0 means no CFG; >1.0 applies classifier-free guidance.
            return_intermediates: if True returns full trajectory per time_grid (like ODESolver).
        Returns:
            final sample (or sequence if return_intermediates=True)
        """

        logging.info(f"Sampling {num_samples} new images... (cond={'yes' if cond is not None else 'no'}, guidance={guidance_scale})")
        model.eval()

        class WrappedModel(ModelWrapper):
            def forward(self, x: torch.Tensor, t: torch.Tensor, **extras):
                batch_size = x.shape[0]
                if t.dim() == 0:
                    t = t.unsqueeze(0).expand(batch_size)
                image = self.model(x, t, **extras)
                return image

        wrapped_model = WrappedModel(model)

        # Build guided velocity callable for ODESolver.
        # ODESolver will call velocity_model(x, t, **extras)
        def guided_velocity(x, t, **extras):
            # extras can contain e.g. 'cond'
            # call wrapped model to obtain velocity under condition
            # mw returns a tensor shaped like x
            if guidance_scale == 1.0:
                return wrapped_model(x, t, **extras)
            else:
                # classifier-free guidance: combine cond & uncond predictions
                # unconditioned prediction: pass cond=None
                # note: ensure no side effects in mw when cond=None
                v_cond = wrapped_model(x, t, **extras)  # uses extras['cond'] if present
                # prepare extras for unconditioned call: drop cond key (or pass cond=None)
                extras_uncond = dict(extras)
                if 'conds' in extras_uncond:
                    extras_uncond['conds'] = None
                v_uncond = wrapped_model(x, t, **extras_uncond)
                return v_uncond + guidance_scale * (v_cond - v_uncond)

        # Time grid (you used 10 steps previously) — keep the same default / user-specified grid if desired
        T = torch.linspace(0, 1, 10)
        T = T.to(device=self.device)

        with torch.no_grad():

            x_init = x_input.to(self.device)
            # Create solver with our guided velocity callable
            solver = ODESolver(velocity_model=guided_velocity)
            # model_extras passed to velocity model at each call; include cond if provided
            model_extras = {}
            if cond is not None:
                model_extras['conds'] = cond.to(self.device)

            sol = solver.sample(time_grid=T, x_init=x_init, method='midpoint',
                                step_size=step_size,
                                return_intermediates=return_intermediates,
                                **model_extras)

            model.train()
            return sol


def train(args):
    setup_logging(args.run_name)
    device = args.device
    train_loader, val_loader = get_train_data(args)
    model = UNet_conditional(cond=args.cond_size).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr)
    l = len(train_loader)
    flow = FLOW_MATCHING(image_size=args.image_size, device=device)
    logger = SummaryWriter(os.path.join("logsw", datetime.now().strftime('%Y%m%d_%H%M') + '_ndata{}'.format(args.train_number)))

    # instantiate an affine path object
    path = AffineProbPath(scheduler=CondOTScheduler())

    for epoch in range(args.epochs):
        logging.info(f"Starting epoch {epoch}:")
        pbar = tqdm(train_loader)
        for i, (images, conds) in enumerate(pbar):
            images_x1 = images.to(device).type(torch.cuda.FloatTensor)
            images_x0 = torch.randn_like(images_x1).to(device)
            conds = conds.to(device).type(torch.cuda.FloatTensor)

            # sample time (user's responsibility)
            t = torch.rand(images_x1.shape[0]).to(device)

            # sample probability path
            path_sample = path.sample(t=t, x_0=images_x0, x_1=images_x1)

            if np.random.random() < 0.1:
               conds = None

            pre_velocity = model(path_sample.x_t, path_sample.t, conds=conds)
            target_velocity = path_sample.dx_t

            # flow matching l2 loss
            loss = torch.pow(pre_velocity - target_velocity, 2).mean()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            pbar.set_postfix(MSE=loss.item())
            logger.add_scalar("MSE", loss.item(), global_step=epoch * l + i)

        if (epoch + 1) % 1 == 0:
            conds = np.array([[-1, -1], [-1, -0.6], [-1, -0.2], [-1, 0.2], [-1, 0.6], [-1, 1],
                              [-0.5, -1], [-0.5, -0.6], [-0.5, -0.2], [-0.5, 0.2], [-0.5, 0.6], [-0.5, 1],
                              [0, -1], [0, -0.6], [0, -0.2], [0, 0.2], [0, 0.6], [0, 1],
                              [0.5, -1], [0.5, -0.6], [0.5, -0.2], [0.5, 0.2], [0.5, 0.6], [0.5, 1],
                              [1, -1], [1, -0.6], [1, -0.2], [1, 0.2], [1, 0.6], [1, 1]])
            conds = torch.from_numpy(conds).float()
            field_gen = flow.sampling(model, 30, conds)
            field_gen = field_gen.cpu().numpy()
            plot_field(field_gen.reshape(30, -1), 'generation_perm'+str(epoch + 1)+'.jpg')

        if (epoch + 1) % 100 == 0:
            torch.save(model.state_dict(), os.path.join("./models", args.run_name, f"ckpt_{epoch + 1}.pt"))
            torch.save(optimizer.state_dict(), os.path.join("./models", args.run_name, f"optim_{epoch + 1}.pt"))


def launch():
    import argparse
    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    args.batch_size = 64
    args.epochs = 2  # 300
    args.image_size = 64
    args.training_data_path = 'perm_15000.h5'
    args.train_number = 15000
    args.cond_size = 2
    args.device = "cuda"
    args.lr = 1e-4
    args.training_rate = 1
    args.run_name = 'FLOW_MATCHING_conditional_' + datetime.now().strftime('%Y%m%d_%H%M') + \
                    '_batchsize{}'.format(args.batch_size) + '_epochs{}'.format(args.epochs) + \
                    '_samples{}'.format(args.train_number)

    train(args)


if __name__ == '__main__':
    launch()
