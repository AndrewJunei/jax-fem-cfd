import jax
import jax.numpy as jnp

""" I, q, and dim index nodes, quadrature points, and spatial dimenstions respectively.
    - coords: nodal coordinates for a given element (I, dim)
    - wq: quadrature weights (q, 1)
    - N: shape function for node I at point q (q, I)
    - Nderiv: derivative wrt dim of shape function for node I at point q (q, dim, I) 
"""

def lin2_shape_functions(coords):
    """ Linear 2-node 1D element
    """
    xi_q = jnp.array([-1, 1]) / jnp.sqrt(3) # (2,)
    N = jnp.stack([(1 - xi_q) / 2, (1 + xi_q) / 2], axis=1) # (2, 2)
    dNdxi = jnp.array([[-0.5, 0.5], [-0.5, 0.5]]) 

    dx = coords[1, 0] - coords[0, 0]  
    dy = coords[1, 1] - coords[0, 1]  
    h = jnp.sqrt(dx**2 + dy**2) # element length

    Nderiv = (dNdxi * (2 / h))[:, jnp.newaxis, :] # (2, 1, 2)
    wq = (jnp.array([1, 1]) * (h / 2))[:, jnp.newaxis] # (2, 1)
    
    return wq, N, Nderiv

def quad4_shape_functions(coords):
    """ Bilinear 4-node quadrilateral element
    """
    xi_q = jnp.array([-1, 1, 1, -1]) / jnp.sqrt(3) # (4,)
    eta_q = jnp.array([-1, -1, 1, 1]) / jnp.sqrt(3) # (4,)

    N = jnp.stack([(1-xi_q)*(1-eta_q), (1+xi_q)*(1-eta_q), 
                   (1+xi_q)*(1+eta_q), (1-xi_q)*(1+eta_q)], axis=1) / 4 # (4, 4)

    dNdxi = jnp.stack([-(1-eta_q), 1-eta_q, 1+eta_q, -(1+eta_q)], axis=1) / 4 # (4, 4)
    dNdeta = jnp.stack([-(1-xi_q), -(1+xi_q), 1+xi_q, 1-xi_q], axis=1) / 4 # (4, 4)
    J = jnp.concatenate([(dNdxi @ coords)[:, jnp.newaxis, :], 
                         (dNdeta @ coords)[:, jnp.newaxis, :]], axis=1) # (4, 2, 2)
    
    wq = jnp.linalg.det(J)[:, jnp.newaxis] # (4, 1)
    Nderiv = jnp.linalg.inv(J) @ jnp.concatenate([dNdxi[:, jnp.newaxis, :], 
                                                  dNdeta[:, jnp.newaxis, :]], axis=1) # (4, 2, 4)
    
    return wq, N, Nderiv

def hex8_shape_functions(coords):
    """ Trilinear 8-node hexahedral element
    """
    xi_q = jnp.array([-1, 1, 1, -1, -1, 1, 1, -1]) / jnp.sqrt(3) # (8,)
    eta_q = jnp.array([-1, -1, 1, 1, -1, -1, 1, 1]) / jnp.sqrt(3) # (8,)
    zeta_q = jnp.array([-1, -1, -1, -1, 1, 1, 1, 1]) / jnp.sqrt(3) # (8,)

    N = jnp.stack([
        (1-xi_q)*(1-eta_q)*(1-zeta_q), (1+xi_q)*(1-eta_q)*(1-zeta_q), 
        (1+xi_q)*(1+eta_q)*(1-zeta_q), (1-xi_q)*(1+eta_q)*(1-zeta_q), 
        (1-xi_q)*(1-eta_q)*(1+zeta_q), (1+xi_q)*(1-eta_q)*(1+zeta_q),
        (1+xi_q)*(1+eta_q)*(1+zeta_q), (1-xi_q)*(1+eta_q)*(1+zeta_q)], axis=1) / 8 # (8, 8)

    dNdxi = jnp.stack([
        -(1-eta_q)*(1-zeta_q), (1-eta_q)*(1-zeta_q), (1+eta_q)*(1-zeta_q), -(1+eta_q)*(1-zeta_q), 
        -(1-eta_q)*(1+zeta_q), (1-eta_q)*(1+zeta_q), (1+eta_q)*(1+zeta_q), -(1+eta_q)*(1+zeta_q)
        ], axis=1) / 8 # (8, 8)
    
    dNdeta = jnp.stack([
        -(1-xi_q)*(1-zeta_q), -(1+xi_q)*(1-zeta_q), (1+xi_q)*(1-zeta_q), (1-xi_q)*(1-zeta_q),
        -(1-xi_q)*(1+zeta_q), -(1+xi_q)*(1+zeta_q), (1+xi_q)*(1+zeta_q), (1-xi_q)*(1+zeta_q)
        ], axis=1) / 8 # (8, 8)
    
    dNdzeta = jnp.stack([
        -(1-xi_q)*(1-eta_q), -(1+xi_q)*(1-eta_q), -(1+xi_q)*(1+eta_q), -(1-xi_q)*(1+eta_q),
        (1-xi_q)*(1-eta_q), (1+xi_q)*(1-eta_q), (1+xi_q)*(1+eta_q), (1-xi_q)*(1+eta_q)
        ], axis=1) / 8 # (8, 8)

    J = jnp.concatenate([(dNdxi @ coords)[:, jnp.newaxis, :], 
                         (dNdeta @ coords)[:, jnp.newaxis, :],
                         (dNdzeta @ coords)[:, jnp.newaxis, :]], axis=1) # (8, 3, 3)

    wq = jnp.linalg.det(J)[:, jnp.newaxis] # (8, 1)

    Nderiv = (jnp.linalg.inv(J) @ 
                jnp.concatenate([dNdxi[:, jnp.newaxis, :],
                                 dNdeta[:, jnp.newaxis, :],
                                 dNdzeta[:, jnp.newaxis, :]], axis=1)) # (8, 3, 8)
    
    return wq, N, Nderiv


def lin3_shape_functions(coords):
    """ Quadratric 3-node 1D element
    """
    xi_q = jnp.array([-jnp.sqrt(3/5), 0.0, jnp.sqrt(3/5)]) 
    wq = jnp.array([5/9, 8/9, 5/9]) 

    N = jnp.stack([xi_q*(xi_q - 1) / 2, 1 - xi_q**2,  xi_q*(xi_q + 1) / 2], axis=1)  # (3, 3)

    dNdxi = jnp.stack([(2*xi_q - 1) / 2, -2*xi_q, (2*xi_q + 1) / 2], axis=1)  # (3, 3)

    dx_dxi = jnp.dot(dNdxi, coords[:, 0])  
    dy_dxi = jnp.dot(dNdxi, coords[:, 1]) 
    J = jnp.sqrt(dx_dxi ** 2 + dy_dxi ** 2)  # (3,)

    Nderiv = (dNdxi / J[:, jnp.newaxis])[:, jnp.newaxis, :]  # (3, 1, 3)
    wq = (wq * J)[:, jnp.newaxis]  # (3, 1)
    
    return wq, N, Nderiv

def quad9_shape_functions(coords):
    """Biquadratic 9-node quadrilateral element
    """
    xi_q_1d = jnp.array([-jnp.sqrt(3/5), 0.0, jnp.sqrt(3/5)])  
    eta_q_1d = jnp.array([-jnp.sqrt(3/5), 0.0, jnp.sqrt(3/5)]) 
    xi_q, eta_q = jnp.meshgrid(xi_q_1d, eta_q_1d)
    xi_q = xi_q.flatten() # (9,)
    eta_q = eta_q.flatten() # (9,)

    wq_1d = jnp.array([5/9, 8/9, 5/9]) 
    wq = jnp.outer(wq_1d, wq_1d).flatten() # (9,)                   

    N = jnp.stack([
        xi_q * eta_q * (xi_q - 1) * (eta_q - 1) / 4,
        (1 - xi_q**2) * eta_q * (eta_q - 1) / 2,
        xi_q * eta_q * (xi_q + 1) * (eta_q - 1) / 4,
        xi_q * (xi_q + 1) * (1 - eta_q**2) / 2,
        xi_q * eta_q * (xi_q + 1) * (eta_q + 1) / 4,
        (1 - xi_q**2) * eta_q * (eta_q + 1) / 2,
        xi_q * eta_q * (xi_q - 1) * (eta_q + 1) / 4,
        xi_q * (xi_q - 1) * (1 - eta_q**2) / 2,
        (1 - xi_q**2) * (1 - eta_q**2)], axis=1) # (9, 9)

    dNdxi = jnp.stack([
        eta_q * (2 * xi_q - 1) * (eta_q - 1) / 4,
        -2 * xi_q * eta_q * (eta_q - 1) / 2,
        eta_q * (2 * xi_q + 1) * (eta_q - 1) / 4,
        (2 * xi_q + 1) * (1 - eta_q**2) / 2,
        eta_q * (2 * xi_q + 1) * (eta_q + 1) / 4,
        -2 * xi_q * eta_q * (eta_q + 1) / 2,
        eta_q * (2 * xi_q - 1) * (eta_q + 1) / 4,
        (2 * xi_q - 1) * (1 - eta_q**2) / 2,
        -2 * xi_q * (1 - eta_q**2)], axis=1) # (9, 9)

    dNdeta = jnp.stack([
        xi_q * (xi_q - 1) * (2 * eta_q - 1) / 4,
        (1 - xi_q**2) * (2 * eta_q - 1) / 2,
        xi_q * (xi_q + 1) * (2 * eta_q - 1) / 4,
        xi_q * (xi_q + 1) * (-2 * eta_q) / 2,
        xi_q * (xi_q + 1) * (2 * eta_q + 1) / 4,
        (1 - xi_q**2) * (2 * eta_q + 1) / 2,
        xi_q * (xi_q - 1) * (2 * eta_q + 1) / 4,
        xi_q * (xi_q - 1) * (-2 * eta_q) / 2,
        -2 * eta_q * (1 - xi_q**2)], axis=1) # (9, 9)

    J = jnp.concatenate([(dNdxi @ coords)[:, jnp.newaxis, :], 
                         (dNdeta @ coords)[:, jnp.newaxis, :]], axis=1) # (9, 2, 2)

    wq = (wq * jnp.linalg.det(J))[:, jnp.newaxis] # (9,)
    Nderiv = jnp.linalg.inv(J) @ jnp.concatenate([dNdxi[:, jnp.newaxis, :], 
                                                  dNdeta[:, jnp.newaxis, :]], axis=1) # (9, 2, 9)

    return wq, N, Nderiv