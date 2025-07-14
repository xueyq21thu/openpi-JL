# src/noise/training/updater.py

import torch
from torch.distributions.kl import kl_divergence

def get_flat_params_from(model: torch.nn.Module) -> torch.Tensor:
    """Helper function to flatten model parameters into a single vector."""
    return torch.cat([param.data.view(-1) for param in model.parameters()])

def set_flat_params_to(model: torch.nn.Module, flat_params: torch.Tensor):
    """Helper function to set model parameters from a flattened vector."""
    prev_ind = 0
    for param in model.parameters():
        flat_size = param.numel()
        param.data.copy_(flat_params[prev_ind:prev_ind + flat_size].view(param.size()))
        prev_ind += flat_size

def conjugate_gradient(fvp, b, n_steps=10, residual_tol=1e-10):
    """
    Computes the solution to Ax = b using the Conjugate Gradient algorithm.
    Here, A is the Fisher Information Matrix, and fvp is a function that
    computes the Fisher-Vector Product (F*v).
    """
    x = torch.zeros_like(b)
    r = b.clone()
    p = b.clone()
    rdotr = torch.dot(r, r)
    for _ in range(n_steps):
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
    states: Dict, # A dict containing state_action_history, text, image
    advantages: torch.Tensor,
    old_log_probs: torch.Tensor,
    max_kl: float = 0.01,
    damping: float = 0.1,
):
    """
    Performs a single TRPO update step for the actor.
    """
    device = next(actor.parameters()).device
    
    # --- Step 1: Compute the policy gradient (g) ---
    with torch.no_grad():
        fixed_dist, _ = actor.get_distribution(**states)
    
    log_probs = fixed_dist.log_prob(torch.round(torch.sigmoid(fixed_dist.logits)))
    ratio = torch.exp(log_probs - old_log_probs)
    policy_loss = -(ratio * advantages).mean()
    
    g = torch.autograd.grad(policy_loss, actor.parameters())
    g_flat = torch.cat([grad.view(-1) for grad in g]).detach()

    # --- Step 2: Define the Fisher-Vector Product (F*v) function ---
    def fisher_vector_product(v):
        kl = kl_divergence(fixed_dist, fixed_dist).mean() # A placeholder KL
        
        grad_kl = torch.autograd.grad(kl, actor.parameters(), create_graph=True)
        grad_kl_flat = torch.cat([grad.view(-1) for grad in grad_kl])

        grad_kl_v = torch.dot(grad_kl_flat, v)
        
        fvp = torch.autograd.grad(grad_kl_v, actor.parameters())
        fvp_flat = torch.cat([grad.view(-1) for grad in fvp]).detach()
        
        return fvp_flat + v * damping

    # --- Step 3: Solve for the search direction (x) using CG ---
    # x = (F^-1)g
    search_dir = conjugate_gradient(fisher_vector_product, g_flat)

    # --- Step 4: Perform line search to find the best step size ---
    shs = 0.5 * torch.dot(search_dir, fisher_vector_product(search_dir))
    lagrange_multiplier = torch.sqrt(shs / max_kl)
    step_size = 1.0 / lagrange_multiplier
    
    final_step_dir = step_size * search_dir
    
    # --- Step 5: Update the actor's parameters ---
    old_params = get_flat_params_from(actor)
    
    # Backtracking line search
    for i in range(10):
        new_params = old_params - final_step_dir * (0.5 ** i)
        set_flat_params_to(actor, new_params)
        
        with torch.no_grad():
            new_dist, _ = actor.get_distribution(**states)
            new_log_probs = new_dist.log_prob(torch.round(torch.sigmoid(new_dist.logits)))
            
            kl = kl_divergence(fixed_dist, new_dist).mean()
            
            new_ratio = torch.exp(new_log_probs - old_log_probs)
            new_loss = -(new_ratio * advantages).mean()
            
            if kl <= max_kl and new_loss <= policy_loss:
                # Accept the update
                return
    
    # If backtracking fails, revert to old parameters
    set_flat_params_to(actor, old_params)