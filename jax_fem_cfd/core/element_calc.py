import jax
import jax.numpy as jnp

""" Currently we assume that a regular mesh is used and rho is constant over each node.
    If this is no longer the case, the functions can be modified to accomodate this by using
    the vmap approach in convection_element_calc.
"""

def lumped_mass_matrix(nconn, wq, N, nn, ndim):
    """ Returns a 1D vector of shape (nn,) containing the diagonal mass matrix entries,
        excluding the factor of rho assumed to be a constant
    """
    Me = jnp.sum((N * wq)[:, :, jnp.newaxis] @ (N[:, jnp.newaxis, :]), axis=0)

    Me_lumped = jnp.sum(Me, axis=1)
    Me_lumped_all = jnp.tile(Me_lumped, nconn.shape[0])
    indices = nconn.flatten()
    M = jax.ops.segment_sum(Me_lumped_all, indices, nn)

    if ndim == 1:
        return M, Me
    elif ndim == 2:
        return jnp.concatenate((M, M)), Me
    elif ndim == 3:
        return jnp.concatenate((M, M, M)), Me

def laplacian_element_calc(wq, Nderiv):
    """ Returns a single laplacian element matrix of shape (nen, nen) 
    """
    Le = jnp.sum((Nderiv * wq[:, :, jnp.newaxis]).transpose(0, 2, 1) @ Nderiv, axis=0)

    return Le

def gradient_element_calc(wq, N, Nderiv, ndim):
    """ Returns Ge for a single element where Ge[i] would build the global G_i
    """
    Ge_list = [jnp.sum((N * wq)[:, :, jnp.newaxis] @ Nderiv[:, i, :][:, jnp.newaxis, :], axis=0)
               for i in range(ndim)] 

    return jnp.stack(Ge_list, axis=0)

def convection_element_calc(nconn, U, wq, N, Nderiv, nn, ndim):
    """ Calculates the convection element matrices
        U = [[u], [v]] or [[u], [v], [w]]. Ce_all has shape (ne, nen, nen)
    """
    def element_convection(ie):
        U_q = N @ jnp.stack([U[nconn[ie] + nn*i] for i in range(ndim)], axis=1) # (q,ndim)
        Ce = jnp.sum((N * wq)[:, :, jnp.newaxis] @ (U_q[:, jnp.newaxis, :] @ Nderiv), axis=0)
        return Ce

    v_element_convection = jax.vmap(element_convection)
    Ce_all = v_element_convection(jnp.arange(nconn.shape[0]))

    return Ce_all

def streamline_diff_element_calc(nconn, node_coords, mu, rho, U, wq, N, Nderiv, nn, ndim):
    """ Calculates the streamline artificial diffusion element matrices that result from
        applying the SUPG test function to the convection term. The parameter tau is given
        by Tezduyar.  
    """
    mu = mu + 1e-10
    
    def element_S(ie):
        U_q = N @ jnp.stack([U[nconn[ie] + nn*i] for i in range(ndim)], axis=1) # (q, ndim)

        U_q_norm = jnp.linalg.norm(U_q, axis=1) # (q,)

        # no longer use constant element length

        U_dir = U_q / (U_q_norm[:, jnp.newaxis] + 1e-10)  # (q, ndim)

        # Get element coordinates (4 nodes for quad)
        elem_coords = node_coords[nconn[ie]]  # (4, 2)
        
        # For quad: nodes 0-1, 1-2, 2-3, 3-0
        edges = jnp.roll(elem_coords, -1, axis=0) - elem_coords  # (4, 2)
        
        proj = jnp.abs(jnp.sum(edges[:, None, :] * U_dir[None, :, :], axis=2))  # (4, q)
        
        # Take max projection for each quad point
        h_elem = jnp.max(proj, axis=0)  # (q,)

        Re = rho * U_q_norm * h_elem / (2 * mu) # (q,)

        tau = jnp.where(Re <= 3, rho * h_elem**2 / (12*mu), h_elem / (2*U_q_norm)) # (q,)
        tau = tau[:, jnp.newaxis, jnp.newaxis] # (q, 1, 1)

        D = U_q[:, :, jnp.newaxis] @ U_q[:, jnp.newaxis, :] # (q, 2, 2)

        Se = jnp.sum((Nderiv * wq[:, :, jnp.newaxis] * tau * rho).transpose(0, 2, 1) 
                     @ (D @ Nderiv), axis=0)
        return Se
    
    v_element_S = jax.vmap(element_S)
    Se_all = v_element_S(jnp.arange(nconn.shape[0]))

    return Se_all

def laplacian_diag(Le, nconn, nn):
    """ Returns the 1D laplacian diagonal entries. Used in Jacobi and EBE preconditioning 
        of the pressure solve 
    """
    Le_diag = jnp.diag(Le)
    Le_diag_all = jnp.tile(Le_diag, nconn.shape[0])

    indices = nconn.flatten()
    L_diag = jax.ops.segment_sum(Le_diag_all, indices, nn)

    return L_diag

def u_dot_n_boundary_integral_1d(bconn, nvec, gamma_d, U_d, nn, node_coords, shapefunc):
    """ Calculates the 1D integral: int_Gamma q (u dot n) dGamma
        bconn: nodes in each boundary element, shape (ne, I)
        nvec: normal vector at each boundary element (assumed constant within each element)
        From gamma_d and U_d, we construct U of shape (ne, I, 2).
    """
    u_indices = jnp.searchsorted(gamma_d, bconn)
    v_indices = jnp.searchsorted(gamma_d, bconn + nn)
    U = jnp.stack([U_d[u_indices], U_d[v_indices]], axis=-1)

    def element_integral(ie):
        wq, N, _ = shapefunc(node_coords[bconn[ie]])

        U_q = N @ U[ie]  # (q, I) x (I, 2) -> (q, 2)
        n_elem = jnp.tile(nvec[ie], (wq.shape[0], 1)) # (q, 2)
        Fe = (N * wq).T @ jnp.sum(n_elem * U_q, axis=1)  # (I,)
        return Fe

    v_element_integral = jax.vmap(element_integral)
    Fe_all = v_element_integral(jnp.arange(bconn.shape[0]))
    indices = bconn.flatten()

    F = jax.ops.segment_sum(Fe_all.flatten(), indices, nn)
    
    return F

def u_dot_n_boundary_integral_2d(bconn, nvec, gamma_d, U_d, nn, node_coords, shapefunc):
    """ Calculates the 2D integral: int_Gamma q (u dot n) dGamma
        bconn: nodes in each boundary element, shape (ne, I)
        nvec: normal vector at each boundary element (assumed constant within each element)
        From gamma_d and U_d, we construct U of shape (ne, I, 3).
    """
    u_indices = jnp.searchsorted(gamma_d, bconn)
    v_indices = jnp.searchsorted(gamma_d, bconn + nn)
    w_indices = jnp.searchsorted(gamma_d, bconn + 2*nn)
    U = jnp.stack([U_d[u_indices], U_d[v_indices], U_d[w_indices]], axis=-1)

    def element_integral(ie):
        # TODO: generalize these coordinates to different faces
        wq, N, _ = shapefunc(node_coords[bconn[ie]][:, [2, 1]])

        U_q = N @ U[ie]  # (q, I) x (I, 3) -> (q, 3)
        n_elem = jnp.tile(nvec[ie], (wq.shape[0], 1)) # (q, 3)
        Fe = (N * wq).T @ jnp.sum(n_elem * U_q, axis=1)  # (I,)
        return Fe

    v_element_integral = jax.vmap(element_integral)
    Fe_all = v_element_integral(jnp.arange(bconn.shape[0]))
    indices = bconn.flatten()

    F = jax.ops.segment_sum(Fe_all.flatten(), indices, nn)
    
    return F

def conservative_convec_elem_calc(nconn, U, wq, N, Nderiv, nn, ndim, beta):
    """ Computes the conservative convection matrix given by Gresho and Sani.
        beta=0 gives the standard advective form, beta=1 gives the divergence form, and
        beta=1/2 gives a skew-symmetric average of the two
    """
    def element_convection(ie):
        U_local = jnp.stack([U[nconn[ie] + nn*i] for i in range(ndim)], axis=1) # (I, ndim)

        U_q = N @ U_local # (q, ndim)
        Ce_A = jnp.sum((N * wq)[:, :, jnp.newaxis] @ (U_q[:, jnp.newaxis, :] @ Nderiv), axis=0)

        divU_q = jnp.sum(jnp.sum(Nderiv * U_local.T, axis=-1), axis=-1)[:, jnp.newaxis] # (q, 1)
        Ce_D = jnp.sum((N * wq * divU_q)[:, :, jnp.newaxis] @ N[:, jnp.newaxis, :], axis=0)

        Ce = Ce_A + beta*Ce_D 
        return Ce

    v_element_convection = jax.vmap(element_convection)
    Ce_all = v_element_convection(jnp.arange(nconn.shape[0]))

    return Ce_all


def Ctau_tilde_elem_calc(nconn, U, wq, N, Nderiv, nn, ndim):
    def element_Ctau_tilde(ie):
        U_q = N @ jnp.stack([U[nconn[ie] + nn*i] for i in range(ndim)], axis=1) # (q, ndim)

        uuT = U_q[:, :, jnp.newaxis] @ U_q[:, jnp.newaxis, :] # (q, 2, 2)

        Ctau_tilde = jnp.sum((Nderiv * wq[:, :, jnp.newaxis]).transpose(0, 2, 1) @ (uuT @ Nderiv), axis=0)
        return Ctau_tilde
    
    v_element_Ke_tilde = jax.vmap(element_Ctau_tilde)
    Ctau_tilde_all = v_element_Ke_tilde(jnp.arange(nconn.shape[0]))

    return Ctau_tilde_all

def calculate_tau(nconn, U, Ce_all, Ctau_tilde_all, N, nn, ndim, D, dt, r):
    D = D + 1e-6 # deal with D=0

    def element_tau(U_elem, Ce, Ctau_tilde):
        # U_elem is (I, ndim)
        U_q = N @ U_elem # (q, ndim)

        Ce_norm = jnp.linalg.norm(Ce)
        Ke_tilde_norm = jnp.linalg.norm(Ctau_tilde)

        U_norm2 = jnp.mean(jnp.sum(U_q**2, axis=1)) 

        Re = (U_norm2 * Ce_norm) / (D * Ke_tilde_norm)

        tau1 = Ce_norm / Ke_tilde_norm
        tau2 = dt / 2
        tau3 = tau1 * Re

        tau = (1/(tau1**r) + 1/(tau2**r) + 1/(tau3**r))**(-1/r)

        return tau
    
    v_element_tau = jax.vmap(element_tau)

    U_elem_all = jnp.stack([U[nconn + nn*i] for i in range(ndim)], axis=-1)

    tau_all = v_element_tau(U_elem_all, Ce_all, Ctau_tilde_all)

    return tau_all