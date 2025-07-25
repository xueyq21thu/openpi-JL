# src/noise/training/updater.py

import torch
from typing import Dict
from tqdm import trange, tqdm
from torch.backends import cudnn
from torch.distributions.kl import kl_divergence

# ==============================================================================
# SECTION 1: UTILITY FUNCTIONS FOR PARAMETER MANIPULATION
# ==============================================================================

def get_flat_params_from(model: torch.nn.Module) -> torch.Tensor:
    """
    Flattens the trainable parameters of a model into a single 1D tensor.
    This is necessary for TRPO's optimization calculations.

    Args:
        model: The PyTorch model (e.g., the Actor).

    Returns:
        A 1D tensor containing all concatenated trainable parameters.
    """
    params = [param.data.view(-1) for param in model.trainable_parameters()]
    if not params:
        return torch.tensor([])
    return torch.cat(params)

def set_flat_params_to(model: torch.nn.Module, flat_params: torch.Tensor):
    """
    Sets the model's trainable parameters from a flattened 1D tensor.
    This is used to update the model's weights during the line search.

    Args:
        model: The PyTorch model to update.
        flat_params: A 1D tensor of parameters to load into the model.
    """
    prev_ind = 0
    for param in model.trainable_parameters():
        flat_size = param.numel()
        param.data.copy_(flat_params[prev_ind:prev_ind + flat_size].view(param.size()))
        prev_ind += flat_size

# ==============================================================================
# SECTION 2: CONJUGATE GRADIENT SOLVER
# ==============================================================================

def conjugate_gradient(fvp, b, n_steps=10, residual_tol=1e-10):
    """
    Computes the solution to Ax = b using the Conjugate Gradient algorithm.
    Includes a tqdm progress bar for visualization.
    """
    x = torch.zeros_like(b)
    r = b.clone()
    p = b.clone()
    rdotr = torch.dot(r, r)
    
    # Wrap the loop in tqdm for a detailed progress bar
    for _ in tqdm(range(n_steps), desc="  - CG Iterations", leave=False, ncols=100):
        fvp_p = fvp(p)
        alpha = rdotr / (torch.dot(p, fvp_p) + 1e-8)
        x += alpha * p
        r -= alpha * fvp_p
        new_rdotr = torch.dot(r, r)
        if new_rdotr < residual_tol:
            break
        beta = new_rdotr / (rdotr + 1e-8)
        p = r + beta * p
        rdotr = new_rdotr
    return x

def trpo_step(
    actor: torch.nn.Module,
    states: Dict,
    advantages: torch.Tensor,
    old_log_probs: torch.Tensor,
    mask_actions: torch.Tensor,
    max_kl: float = 0.01,
    damping: float = 0.1,
):
    """
    Performs a single, constrained policy update using the TRPO algorithm.
    The progress bar is now more granular.
    """
    with cudnn.flags(enabled=False):
        # --- Step 1: Compute Policy Gradient (outside of progress bar, it's fast) ---
        dist, _ = actor.get_distribution(**states)
        log_probs = dist.log_prob(mask_actions)
        ratio = torch.exp(log_probs - old_log_probs)
        policy_loss = -(ratio * advantages).mean()
        g = torch.autograd.grad(policy_loss, actor.trainable_parameters(), retain_graph=True)
        g_flat = torch.cat([grad.view(-1) for grad in g]).detach()

        with torch.no_grad():
            fixed_dist, _ = actor.get_distribution(**states)
        
        def fisher_vector_product(v):
            # ... (this inner function is unchanged) ...
            dist_for_fvp, _ = actor.get_distribution(**states)
            kl = kl_divergence(fixed_dist, dist_for_fvp).mean()
            grad_kl = torch.autograd.grad(kl, actor.trainable_parameters(), create_graph=True)
            grad_kl_flat = torch.cat([grad.view(-1) for grad in grad_kl])
            grad_kl_v = torch.dot(grad_kl_flat, v)
            fvp = torch.autograd.grad(grad_kl_v, actor.trainable_parameters())
            fvp_flat = torch.cat([grad.view(-1) for grad in fvp]).detach()
            return fvp_flat + v * damping

        # --- Step 2: Solve for Search Direction (this is slow, so we visualize it) ---
        print("  -> Solving for search direction using Conjugate Gradient...")
        search_dir = conjugate_gradient(fisher_vector_product, g_flat) # This will now show its own bar

        # --- Step 3: Compute Step Size and Perform Line Search ---
        print("  -> Performing backtracking line search...")
        shs = 0.5 * torch.dot(search_dir, fisher_vector_product(search_dir))
        lagrange_multiplier = torch.sqrt(shs / (max_kl + 1e-8))
        step_size = 1.0 / lagrange_multiplier
        final_step_dir = step_size * search_dir
        
        old_params = get_flat_params_from(actor)
        initial_loss = policy_loss.item()

        # Wrap the line search loop in trange for a progress bar
        for i in trange(10, desc="  - Line Search Steps", leave=False, ncols=100):
            new_params = old_params - final_step_dir * (0.5 ** i)
            set_flat_params_to(actor, new_params)
            
            with torch.no_grad():
                new_dist, _ = actor.get_distribution(**states)
                kl = kl_divergence(fixed_dist, new_dist).mean()
                new_log_probs = new_dist.log_prob(mask_actions)
                new_ratio = torch.exp(new_log_probs - old_log_probs)
                new_loss = -(new_ratio * advantages).mean()
                
                # Update the progress bar's postfix with live info
                tqdm.write(f"     Step {i}: KL={kl.item():.6f}, Loss={new_loss.item():.4f}")

                if kl <= max_kl and new_loss < initial_loss:
                    print(f"  -> Update accepted at line search step {i}.")
                    return
        
        print("  -> Line search failed. Reverting to old actor parameters.")
        set_flat_params_to(actor, old_params)

# ==============================================================================
# SECTION 3: THE TRPO UPDATE STEP
# ==============================================================================

def trpo_step(
    actor: torch.nn.Module,
    states: Dict,
    advantages: torch.Tensor,
    old_log_probs: torch.Tensor,
    mask_actions: torch.Tensor,
    max_kl: float = 0.01,
    damping: float = 0.1,
):
    """
    Performs a single, constrained policy update using the TRPO algorithm.
    Includes a progress bar for visual feedback.
    """
    # --- MODIFICATION: Setup tqdm progress bar ---
    # We define the major steps that will be tracked by the progress bar.

    
    # To enable double backward for the Fisher-Vector Product with RNNs,
    # we must disable the highly optimized but limited CuDNN backend.
    with cudnn.flags(enabled=False):
        
        # --- Step 1: Compute the Policy Gradient (g) ---
        dist, _ = actor.get_distribution(**states)
        log_probs = dist.log_prob(mask_actions)
        ratio = torch.exp(log_probs - old_log_probs)
        policy_loss = -(ratio * advantages).mean()
        g = torch.autograd.grad(policy_loss, actor.trainable_parameters(), retain_graph=True)
        g_flat = torch.cat([grad.view(-1) for grad in g]).detach()


        # --- Step 2: Define and Compute Fisher-Vector Product & Search Direction ---
        with torch.no_grad():
            fixed_dist, _ = actor.get_distribution(**states)
        
        def fisher_vector_product(v):
            dist_for_fvp, _ = actor.get_distribution(**states)
            kl = kl_divergence(fixed_dist, dist_for_fvp).mean()
            grad_kl = torch.autograd.grad(kl, actor.trainable_parameters(), create_graph=True)
            grad_kl_flat = torch.cat([grad.view(-1) for grad in grad_kl])
            grad_kl_v = torch.dot(grad_kl_flat, v)
            fvp = torch.autograd.grad(grad_kl_v, actor.trainable_parameters())
            fvp_flat = torch.cat([grad.view(-1) for grad in fvp]).detach()
            return fvp_flat + v * damping

        print("  -> Solving for search direction using Conjugate Gradient...")
        search_dir = conjugate_gradient(fisher_vector_product, g_flat) # This will now show its own bar

        # --- Step 3: Compute Step Size via Line Search ---
        print("  -> Performing backtracking line search...")
        shs = 0.5 * torch.dot(search_dir, fisher_vector_product(search_dir))
        lagrange_multiplier = torch.sqrt(shs / (max_kl + 1e-8))
        step_size = 1.0 / lagrange_multiplier
        final_step_dir = step_size * search_dir
        
        old_params = get_flat_params_from(actor)
        initial_loss = policy_loss.item()

        # Wrap the line search loop in trange for a progress bar
        for i in trange(10, desc="  - Line Search Steps", leave=False, ncols=100):
            new_params = old_params - final_step_dir * (0.5 ** i)
            set_flat_params_to(actor, new_params)
            
            with torch.no_grad():
                new_dist, _ = actor.get_distribution(**states)
                kl = kl_divergence(fixed_dist, new_dist).mean()
                new_log_probs = new_dist.log_prob(mask_actions)
                new_ratio = torch.exp(new_log_probs - old_log_probs)
                new_loss = -(new_ratio * advantages).mean()
                
                # Update the progress bar's postfix with live info
                tqdm.write(f"     Step {i}: KL={kl.item():.6f}, Loss={new_loss.item():.4f}")

                if kl <= max_kl and new_loss < initial_loss:
                    print(f"  -> Update accepted at line search step {i}.")
                    return
        
        print("  -> Line search failed. Reverting to old actor parameters.")
        set_flat_params_to(actor, old_params)

def ppo_step(
    actor: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    states: Dict,
    advantages: torch.Tensor,
    old_log_probs: torch.Tensor,
    mask_actions: torch.Tensor,
    ppo_epochs: int = 10,
    clip_param: float = 0.2,
):
    """
    Performs multiple PPO update steps for the actor using a single batch of data.
    Includes a tqdm progress bar to visualize the update epochs.

    Args:
        actor: The actor model to be updated.
        optimizer: The optimizer for the actor model.
        states: A dictionary containing the batched state information (history, image, text).
        advantages: The GAE computed for each step in the batch.
        old_log_probs: The log probabilities of the actions taken, from the policy *before* the update.
        mask_actions: The actual binary actions (0 or 1) taken in the batch.
        ppo_epochs: The number of optimization epochs to perform on the data.
        clip_param: The clipping parameter epsilon for the PPO objective.
    """
    # --- MODIFICATION: Wrap the PPO epochs loop with trange ---
    # We use trange, which is a specialized version of tqdm for `range`.
    # `leave=False` makes the progress bar disappear after it's done,
    # keeping the console log tidy.
    for _ in trange(ppo_epochs, desc="  - PPO Update Epochs", leave=False):
        # --- 1. Re-evaluate the actions with the current policy ---
        # This needs to be done in every PPO epoch because the policy `actor` is being updated.
        dist, _ = actor.get_distribution(**states)
        new_log_probs = dist.log_prob(mask_actions)
        
        # --- 2. Calculate the ratio and the surrogate loss ---
        # The ratio r_t(θ) = π_θ(a_t|s_t) / π_θ_old(a_t|s_t)
        ratio = torch.exp(new_log_probs - old_log_probs)
        
        # The first part of the surrogate objective L_CPI(θ)
        surr1 = ratio * advantages
        
        # The second, "clipped" part of the objective
        surr2 = torch.clamp(ratio, 1.0 - clip_param, 1.0 + clip_param) * advantages
        
        # --- 3. The PPO loss ---
        # The final loss is the element-wise minimum of the two surrogate objectives,
        # averaged over the batch. We take the negative because optimizers minimize.
        actor_loss = -torch.min(surr1, surr2).mean()
        
        # --- 4. Standard optimization step ---
        optimizer.zero_grad()
        actor_loss.backward()
        optimizer.step()