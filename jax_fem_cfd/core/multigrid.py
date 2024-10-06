import jax
import jax.numpy as jnp
from functools import partial
from ..core import element_calc as ecalc
from ..core.element_mult import laplacian_element_mult
from ..utils.jax_iterative_solvers import cg


def multigrid_setup(num_elem, num_elemf, domain_size, outlet_faces, mesh_func, shapefunc):
    """ setup_func takes inputs ne_x, ne_y and 
        outputs nn, nnx, nny, nconn, gamma_p, Le, Le_no_diag, L_diag
    """
    nex, ney = num_elem
    N = int(jnp.log2(nex // num_elemf[0])) + 1
    
    lvl_nums = jnp.zeros((N + 1, 5), dtype=int) # [nn, nnx, nny, ne, np]
    # pad with zeros at first entry, nn, ne, and np are cumulative
    Le_lvls = jnp.zeros((N, 2, 4, 4)) # (N, 2, nen, nen), [Le, Le_off]

    nconn_lvls, L_diag_lvls, gamma_p_lvls = [], [], []

    for i in range(N):
        node_coords, nconn, surfnodes, _, _ = mesh_func(nex, ney, *domain_size)
        wq, _, Nderiv = shapefunc(node_coords[nconn[0]])
        nn = node_coords.shape[0]
        nnx, nny = nex + 1, ney + 1 # only for linear shape functions

        if not outlet_faces: 
            gamma_p = jnp.array([0]) 
        else:
            gamma_p = jnp.unique(jnp.concatenate([surfnodes[i] for i in outlet_faces])) 

        Le = ecalc.laplacian_element_calc(wq, Nderiv)
        L_diag = ecalc.laplacian_diag(Le, nconn, nn)
        Le_off = Le - jnp.diag(jnp.diag(Le))
        
        total_nn = lvl_nums[i, 0] + nn
        total_ne = lvl_nums[i, 3] + nex*ney
        nums = jnp.array([total_nn, nnx, nny, total_ne, lvl_nums[i, 4] + gamma_p.shape[0]])
        lvl_nums = lvl_nums.at[i + 1].set(nums)

        Le_lvls = Le_lvls.at[i, 0].set(Le)
        Le_lvls = Le_lvls.at[i, 1].set(Le_off)

        nconn_lvls.append(nconn)
        L_diag_lvls.append(L_diag)
        gamma_p_lvls.append(gamma_p)

        nex, ney = nex // 2, ney // 2 # next mesh

    nconn_lvls = jnp.concatenate(nconn_lvls)
    L_diag_lvls = jnp.concatenate(L_diag_lvls)
    gamma_p_lvls = jnp.concatenate(gamma_p_lvls)

    lvl_nums = tuple(tuple(int(x) for x in row) for row in lvl_nums) # convert to tuple

    return lvl_nums, Le_lvls, nconn_lvls, L_diag_lvls, gamma_p_lvls

@partial(jax.jit, static_argnums=(1, 2, 3, 4))
def interpolation_operator(xH, nnH_x, nnH_y, nnh_x, nnh_y):
    """ Takes x vector xH of shape (nnH,) on a coarse 2D regular mesh and performs bilinear 
        interpolation to map it to xh of shape (nnh,) on a fine 2D regular mesh where the 
        coarse element length is twice that of the fine element length. All indexing is done
        with the conventions given in the create_mesh_2d function.
    """
    xh2d = jnp.zeros((nnh_y, nnh_x))
    xH2d = xH.reshape(nnH_y, nnH_x)

    ih_even = jnp.arange(0, nnh_y, 2)
    ih_odd = jnp.arange(1, nnh_y - 1, 2)
    jh_even = jnp.arange(0, nnh_x, 2)
    jh_odd = jnp.arange(1, nnh_x - 1, 2)

    iH = ih_even // 2 # coarse i indices
    jH = jh_even // 2 # coarse j indices

    left_jH = jH[:-1] # exclude right most point
    right_jH = jH[1:] # exclude left most point
    bottom_iH = iH[:-1] # exclude top most
    top_iH = iH[1:] # exclude bottom most

    # even ih, even jh
    xh2d = xh2d.at[ih_even[:, jnp.newaxis], jh_even].set(xH2d[iH[:, jnp.newaxis], jH]) 

    # even ih, odd jh
    xh2d = xh2d.at[ih_even[:, jnp.newaxis], jh_odd].set(0.5 * xH2d[iH[:, jnp.newaxis], left_jH] 
                                    + 0.5 * xH2d[iH[:, jnp.newaxis], right_jH])

    # odd ih, even jh
    xh2d = xh2d.at[ih_odd[:, jnp.newaxis], jh_even].set(0.5 * xH2d[bottom_iH[:, jnp.newaxis], jH] 
                                    + 0.5 * xH2d[top_iH[:, jnp.newaxis], jH])

    # odd ih, odd jh
    xh2d = xh2d.at[ih_odd[:, jnp.newaxis], jh_odd].set(0.25 * xH2d[bottom_iH[:, jnp.newaxis], left_jH] 
                                    + 0.25 * xH2d[bottom_iH[:, jnp.newaxis], right_jH]
                                    + 0.25 * xH2d[top_iH[:, jnp.newaxis], left_jH]
                                    + 0.25 * xH2d[top_iH[:, jnp.newaxis], right_jH])
    
    return xh2d.flatten()

@partial(jax.jit, static_argnums=(1, 2, 3, 4))
def restriction_operator(xh, nnh_x, nnh_y, nnH_x, nnH_y):
    """ Takes a vector xh on the fine mesh and maps it to a vector on the coarse mesh by using
        the Galerkin approach. If interpolation_operator(xH) performs I @ xH where I is
        the interpolation matrix, then restriction_operator(xh) performs R @ xh where the 
        restriction operator is R = I^T.
    """
    xh2d = xh.reshape(nnh_y, nnh_x)
    xH2d = jnp.zeros((nnH_y, nnH_x))

    iH = jnp.arange(nnH_y)
    jH = jnp.arange(nnH_x)

    vert_ih = 2 * iH[1:-1] # used for left and right edges
    horiz_jh = 2 * jH[1:-1] # used for top and bottom edges

    # for each coarse (iH, jH), the corresponding fine indices are (ih, jh) = (2*iH, 2*jH)

    xH2d = xH2d.at[0, 0].set(xh2d[0, 0] # bottom left corner
                    + 0.5*(xh2d[0, 1] + xh2d[1, 0]) 
                    + 0.25*xh2d[1, 1]) 
    xH2d = xH2d.at[0, nnH_x-1].set(xh2d[0, nnh_x-1] # bottom right corner
                        + 0.5*(xh2d[0, nnh_x-2] + xh2d[1, nnh_x-1]) 
                        + 0.25*xh2d[1, nnh_x-2])
    xH2d = xH2d.at[nnH_y-1, 0].set(xh2d[nnh_y-1, 0] # top left corner
                        + 0.5*(xh2d[nnh_y-2, 0] + xh2d[nnh_y-1, 1]) 
                        + 0.25*xh2d[nnh_y-2, 1])
    xH2d = xH2d.at[nnH_y-1, nnH_x-1].set(xh2d[nnh_y-1, nnh_x-1] # top right corner
                                + 0.5*(xh2d[nnh_y-1, nnh_x-2] + xh2d[nnh_y-2, nnh_x-1]) 
                                + 0.25*xh2d[nnh_y-2, nnh_x-2])

    xH2d = xH2d.at[iH[1:-1], 0].set(xh2d[vert_ih, 0] # left edge
                        + 0.5*(xh2d[vert_ih+1, 0] + xh2d[vert_ih-1, 0] + xh2d[vert_ih, 1])
                        + 0.25*(xh2d[vert_ih+1, 1] + xh2d[vert_ih-1, 1]))

    xH2d = xH2d.at[iH[1:-1], nnH_x-1].set(xh2d[vert_ih, nnh_x-1] # right edge
                            + 0.5*(xh2d[vert_ih+1, nnh_x-1] + xh2d[vert_ih-1, nnh_x-1]
                                    + xh2d[vert_ih, nnh_x-2])
                            + 0.25*(xh2d[vert_ih+1, nnh_x-2] + xh2d[vert_ih-1, nnh_x-2]))

    xH2d = xH2d.at[0, jH[1:-1]].set(xh2d[0, horiz_jh] # bottom edge
                        + 0.5*(xh2d[0, horiz_jh+1] + xh2d[0, horiz_jh-1] + xh2d[1, horiz_jh])
                        + 0.25*(xh2d[1, horiz_jh+1] + xh2d[1, horiz_jh-1]))

    xH2d = xH2d.at[nnH_y-1, jH[1:-1]].set(xh2d[nnh_y-1, horiz_jh] # top edge
                            + 0.5*(xh2d[nnh_y-1, horiz_jh+1] + xh2d[nnh_y-1, horiz_jh-1]
                                    + xh2d[nnh_y-2, horiz_jh])
                            + 0.25*(xh2d[nnh_y-2, horiz_jh+1] + xh2d[nnh_y-2, horiz_jh-1]))

    # interior points
    xH2d = xH2d.at[iH[1:-1, jnp.newaxis], jH[1:-1]].set(
        xh2d[vert_ih[:, jnp.newaxis], horiz_jh]
        + 0.5 * (xh2d[vert_ih[:, jnp.newaxis]+1, horiz_jh] + xh2d[vert_ih[:, jnp.newaxis]-1, horiz_jh] 
                 + xh2d[vert_ih[:, jnp.newaxis], horiz_jh+1] + xh2d[vert_ih[:, jnp.newaxis], horiz_jh-1])
        + 0.25 * (xh2d[vert_ih[:, jnp.newaxis]+1, horiz_jh+1] + xh2d[vert_ih[:, jnp.newaxis]+1, horiz_jh-1] 
                  + xh2d[vert_ih[:, jnp.newaxis]-1, horiz_jh+1] + xh2d[vert_ih[:, jnp.newaxis]-1, horiz_jh-1])
        )
    
    # full weighting would divide by a factor of 4 - this works very poorly with the approach of 
    # discretizing again at each level - for us R = I^T
    return xH2d.flatten() 

def single_jacobi_iter(x, rhs, Le_off, L_diag, gamma_p, nconn, nn, omega=2/3):
    x = x.at[gamma_p].set(0)  # modify initial guess
    off_diag = laplacian_element_mult(x, Le_off, nconn, nn)
    x = (omega * (rhs - off_diag) / L_diag) + (1-omega) * x
    return x.at[gamma_p].set(0)


@partial(jax.jit, static_argnames=("lvl_nums"))
def v_cycle(rhs, lvl_nums, Le_lvls, nconn_lvls, L_diag_lvls, gamma_p_lvls):
    # solve L @ x1 = rhs, and assume the initial guess x0 = 0
    # rhs[gamma_p] = 0 from the cg algorithm
    N = len(lvl_nums) - 1
    x_lvls = jnp.zeros(lvl_nums[-1][0]) # x1, x2, ..., xN
    r_lvls = jnp.zeros(lvl_nums[-1][0]).at[0:lvl_nums[1][0]].set(rhs) # r1 = rhs, r2, ..., rN

    # smooth x1, ..., x{N-1} and restrict to r2, ..., rN
    for n in range(N - 1):
        nconnh = nconn_lvls[lvl_nums[n][3]:lvl_nums[n+1][3]]
        gamma_ph = gamma_p_lvls[lvl_nums[n][4]:lvl_nums[n+1][4]]
        nnh = lvl_nums[n+1][0] - lvl_nums[n][0]

        rh = r_lvls[lvl_nums[n][0]:lvl_nums[n+1][0]].at[gamma_ph].set(0) # not 100% necessary 
        
        # when 1 jacobi iter is used, we can do this instead for efficiency
        xh = (2/3 * rh / L_diag_lvls[lvl_nums[n][0]:lvl_nums[n+1][0]]).at[gamma_ph].set(0)
    
        rH = rh - (laplacian_element_mult(xh, Le_lvls[n, 0], nconnh, nnh).at[gamma_ph].set(0)) 
        
        rH = restriction_operator(rH, lvl_nums[n+1][1], lvl_nums[n+1][2],
                                      lvl_nums[n+2][1], lvl_nums[n+2][2])
    
        x_lvls = x_lvls.at[lvl_nums[n][0]:lvl_nums[n+1][0]].set(xh)
        r_lvls = r_lvls.at[lvl_nums[n+1][0]:lvl_nums[n+2][0]].set(rH)

    # solve for xN
    nconn_N = nconn_lvls[lvl_nums[N-1][3]:lvl_nums[N][3]]
    gamma_p_N = gamma_p_lvls[lvl_nums[N-1][4]:lvl_nums[N][4]]
    Le_N = Le_lvls[N-1, 0]
    nn_N = lvl_nums[N][0] - lvl_nums[N-1][0]

    def cg_operator(x):
        result = laplacian_element_mult(x, Le_N, nconn_N, nn_N)
        return result.at[gamma_p_N].set(0)

    rN = r_lvls[lvl_nums[N-1][0]:lvl_nums[N][0]].at[gamma_p_N].set(0) # should do this
    xN, _ = cg(cg_operator, rN, tol=1e-6)

    x_lvls = x_lvls.at[lvl_nums[N-1][0]:lvl_nums[N][0]].set(xN)

    # correct and smooth x{N-1}, ..., x1
    for n in range(N - 2, -1, -1):
        nconnh = nconn_lvls[lvl_nums[n][3]:lvl_nums[n+1][3]]
        gamma_ph = gamma_p_lvls[lvl_nums[n][4]:lvl_nums[n+1][4]]
        nnh = lvl_nums[n+1][0] - lvl_nums[n][0]

        rh = r_lvls[lvl_nums[n][0]:lvl_nums[n+1][0]]
        xH = x_lvls[lvl_nums[n+1][0]:lvl_nums[n+2][0]]
        xh = x_lvls[lvl_nums[n][0]:lvl_nums[n+1][0]]

        xH = interpolation_operator(xH, lvl_nums[n+2][1], lvl_nums[n+2][2],
                                        lvl_nums[n+1][1], lvl_nums[n+1][2])

        xh = xh + xH
        # xh = jacobi_smooth(xh, rh,
        #                    Le_lvls[n, 1], L_diag_lvls[lvl_nums[n][0]:lvl_nums[n+1][0]],
        #                    gamma_ph, nconnh, nnh, iters=1)
        
        xh = single_jacobi_iter(xh, rh,
                                Le_lvls[n, 1], L_diag_lvls[lvl_nums[n][0]:lvl_nums[n+1][0]],
                                gamma_ph, nconnh, nnh)
    
        x_lvls = x_lvls.at[lvl_nums[n][0]:lvl_nums[n+1][0]].set(xh)

    x1 = x_lvls[0:lvl_nums[1][0]] # from last iter of the loop

    return x1