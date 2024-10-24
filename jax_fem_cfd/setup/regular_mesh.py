import jax
import jax.numpy as jnp
from functools import partial

@partial(jax.jit, static_argnums=(0, 1, 2, 3))
def create_mesh_2d(num_elem_x, num_elem_y, domain_size_x, domain_size_y):
    """ 2D, rectangular domain, regular mesh
        - bottom left corner is node 0, local (i, j) -> global j*nnx + i
        - node_coords: x and y coords for each node in the mesh
        - nconn: node indices contained within each element
        - surfnodes: node indices on the boundary faces [down, right, top, left]
        - bconns: node indices contained within each boundary element for each boundary
        - nvecs: normal vectors of each boundary element for each boundary
    """
    nnx, nny = num_elem_x + 1, num_elem_y + 1
    dx = domain_size_x / num_elem_x
    dy = domain_size_y / num_elem_y

    def compute_node_coords(i, j):
        return jnp.array([i * dx, j * dy])

    def compute_nconn(ie, je):
        return jnp.array([je * nnx + ie, je * nnx + ie + 1, 
                          (je + 1) * nnx + ie + 1, (je + 1) * nnx + ie], dtype=int)

    node_coords = jax.vmap(jax.vmap(compute_node_coords, in_axes=(0, None)), in_axes=(None, 0))(
                            jnp.arange(nnx), jnp.arange(nny)).reshape(-1, 2)

    nconn = jax.vmap(jax.vmap(compute_nconn, in_axes=(0, None)), in_axes=(None, 0))(
                        jnp.arange(num_elem_x), jnp.arange(num_elem_y)).reshape(-1, 4)

    surfnodesD = jnp.arange(nnx)
    surfnodesR = nnx * jnp.arange(1, nny + 1) - 1
    surfnodesT = (nnx * (nny - 1)) + jnp.arange(0, nnx)
    surfnodesL = nnx * jnp.arange(nny)
    surfnodes = [surfnodesD, surfnodesR, surfnodesT, surfnodesL]

    bconns = [jnp.stack([surfnodes[i][:-1], surfnodes[i][1:]], axis=1) for i in range(4)]

    normals = jnp.array([[0, -1], [1, 0], [0, 1], [-1, 0]])
    nvecs = [jnp.tile(normals[i], (bconns[i].shape[0], 1)) for i in range(4)]

    return node_coords, nconn, surfnodes, bconns, nvecs

@partial(jax.jit, static_argnums=(0, 1, 2, 3, 4, 5))
def create_mesh_3d(ne_x, ne_y, ne_z, domain_size_x, domain_size_y, domain_size_z):
    """ 3D, cuboid domain, regular mesh
        local node (i, j, k) -> global node k*nnx*nny + j*nnx + i 
        Note that in my routines (mesh and shape functions), positive z-axis is opposite of what
        it would be from the right hand rule. This was an accident, but it does not matter.
        - node_coords: x, y, z coords for each node in the mesh
        - nconn: node indices contained in each element
        - surfnodes: node indices on the boundary faces [down, right, top, left, front, back]
        - bconns: node indices contained within each boundary element for each boundary
        - nvecs: normal vectors of each boundary element for each boundary
    """
    nnx, nny, nnz = ne_x + 1, ne_y + 1, ne_z + 1
    dx, dy, dz = domain_size_x / ne_x, domain_size_y / ne_y, domain_size_z / ne_z

    def compute_node_coords(i, j, k):
        return jnp.array([i*dx, j*dy, k*dz])

    def compute_nconn(ie, je, ke):
        nodes_2d = jnp.array([je*nnx + ie, je*nnx + ie + 1, 
                              (je + 1)*nnx + ie + 1, (je + 1)*nnx + ie], dtype=int)
        return jnp.concatenate([ke*nnx*nny + nodes_2d, (ke + 1)*nnx*nny + nodes_2d])

    node_coords = jax.vmap(jax.vmap(jax.vmap(compute_node_coords, 
                    in_axes=(0, None, None)), in_axes=(None, 0, None)), in_axes=(None, None, 0))(
                        jnp.arange(nnx), jnp.arange(nny), jnp.arange(nnz)).reshape(-1, 3)

    nconn = jax.vmap(jax.vmap(jax.vmap(compute_nconn, 
                in_axes=(0, None, None)), in_axes=(None, 0, None)), in_axes=(None, None, 0))(
                    jnp.arange(ne_x), jnp.arange(ne_y), jnp.arange(ne_z)).reshape(-1, 8)

    nconn_3d = nconn.reshape(ne_z, ne_y, ne_x, 8)
    bconn_D = nconn_3d[:, 0, :, [0, 1, 5, 4]].reshape(-1, 4)
    bconn_R = nconn_3d[:, :, -1, [1, 5, 6, 2]].reshape(-1, 4)
    bconn_T = nconn_3d[:, -1, :, [3, 2, 6, 7]].reshape(-1, 4)
    bconn_L = nconn_3d[:, :, 0, [0, 4, 7, 3]].reshape(-1, 4)
    bconn_F = nconn_3d[0, :, :, [0, 1, 2, 3]].reshape(-1, 4)
    bconn_B = nconn_3d[-1, :, :, [4, 5, 6, 7]].reshape(-1, 4)

    bconns = [bconn_D, bconn_R, bconn_T, bconn_L, bconn_F, bconn_B]

    # surfnodes = [jnp.unique(bconns[i]) for i in range(6)] # issue with jit

    surfnodesD = (jnp.tile(jnp.arange(nnx), nnz) + 
                  jnp.repeat(jnp.arange(nnz) * nnx * nny, nnx))  
    surfnodesR = (jnp.tile(nnx * (jnp.arange(1, nny + 1)) - 1, nnz) + 
                  jnp.repeat(jnp.arange(nnz) * nnx * nny, nny))  
    surfnodesT = (jnp.tile(nnx * (nny - 1) + jnp.arange(nnx), nnz) + 
                  jnp.repeat(jnp.arange(nnz) * nnx * nny, nnx))  
    surfnodesL = (jnp.tile(nnx * jnp.arange(nny), nnz) + 
                  jnp.repeat(jnp.arange(nnz) * nnx * nny, nny))
    surfnodesF = jnp.arange(nnx * nny)  # first 2D slice
    surfnodesB = nnx * nny * (nnz - 1) + jnp.arange(nnx * nny)  # last 2D slice
    surfnodes = [surfnodesD, surfnodesR, surfnodesT, surfnodesL, surfnodesF, surfnodesB]

    # z-axis is actually flipped from standard convention 
    normals = jnp.array([[0, -1, 0], [1, 0, 0], [0, 1, 0], 
                         [-1, 0, 0], [0, 0, -1],[0, 0, 1]]) 
    nvecs = [jnp.tile(normals[i], (bconns[i].shape[0], 1)) for i in range(6)]
    
    return node_coords, nconn, surfnodes, bconns, nvecs

@partial(jax.jit, static_argnums=(0, 1, 2, 3))
def create_mesh_2d_quad9(nex, ney, Lx, Ly):
    """ 2D regular mesh for 9-node quadrilateral elements
    """
    nnx, nny = 2*nex + 1, 2*ney + 1
    dx, dy = Lx / nex, Ly / ney

    def compute_node_coords(i, j):
        return jnp.array([i*dx/2, j*dy/2])

    def compute_nconn(ie, je):
        return jnp.array([(2*ie) + 2*je*nnx, (2*ie + 1) + 2*je*nnx, 
                          (2*ie + 2) + 2*je*nnx, (2*ie + 2) + (2*je + 1)*nnx, 
                          (2*ie + 2) + (2*je + 2)*nnx, (2*ie + 1) + (2*je + 2)*nnx,
                          (2*ie) + (2*je + 2)*nnx, (2*ie) + (2*je + 1)*nnx,
                          (2*ie + 1) + (2*je + 1)*nnx], dtype=int)
    
    node_coords = jax.vmap(jax.vmap(compute_node_coords, in_axes=(0, None)), 
                           in_axes=(None, 0))(jnp.arange(nnx), jnp.arange(nny)).reshape(-1, 2)

    elem_indices = jnp.arange(nex * ney)
    ie_indices = elem_indices % nex  # element index in x-direction
    je_indices = elem_indices // nex  # element index in y-direction

    nconn = jax.vmap(compute_nconn)(ie_indices, je_indices)

    surfnodesD = jnp.arange(0, nnx) 
    surfnodesR = jnp.arange(nnx - 1, nnx * nny, nnx)
    surfnodesT = jnp.arange(nnx * (nny - 1), nnx * nny) # can index backward
    surfnodesL = jnp.arange(0, nnx * nny, nnx) # can index backward 

    surfnodes = [surfnodesD, surfnodesR, surfnodesT, surfnodesL]
    
    bconns = [jnp.stack([nodes[:-2:2], nodes[1:-1:2], nodes[2::2]], axis=1) for nodes in surfnodes]
    
    normals = jnp.array([[0, -1], [1, 0], [0, 1], [-1, 0]])
    nvecs = [jnp.tile(normals[i], (bconns[i].shape[0], 1)) for i in range(4)]

    return node_coords, nconn, surfnodes, bconns, nvecs

# TODO: Add create_mesh_3d_hex27 here later

@partial(jax.jit, static_argnums=(0, 1, 2, 3))
def create_periodic_mesh_2d(nex, ney, Lx, Ly):
    """ 2D quad4 regular mesh with periodic boundary conditions
    """
    nnx, nny = nex, ney # excluding periodic nodes
    dx = Lx / nex
    dy = Ly / ney

    def compute_node_coords(i, j):
        return jnp.array([i * dx, j * dy])

    node_coords = jax.vmap(jax.vmap(compute_node_coords, in_axes=(0, None)), in_axes=(None, 0))(
        jnp.arange(nnx), jnp.arange(nny)).reshape(-1, 2)

    def compute_nconn(ie, je):
        n0 = ie + je * nnx
        n1 = (ie + 1) % nnx + je * nnx # wrap around
        n2 = (ie + 1) % nnx + ((je + 1) % nny) * nnx # wrap around
        n3 = ie + ((je + 1) % nny) * nnx # wrap around
        return jnp.array([n0, n1, n2, n3], dtype=int)

    elem_indices = jnp.arange(nex * ney)
    ie_indices = elem_indices % nex
    je_indices = elem_indices // nex

    nconn = jax.vmap(compute_nconn)(ie_indices, je_indices)

    surfnodes = None
    bconns = None
    nvecs = None

    return node_coords, nconn, surfnodes, bconns, nvecs

@partial(jax.jit, static_argnums=(0, 1, 2, 3))
def create_periodic_mesh_2d_quad9(nex, ney, Lx, Ly):
    """ 2D quad9 regular mesh with periodic boundary conditions
    """
    nnx, nny = 2 * nex, 2 * ney # excluding periodic nodes
    dx, dy = Lx / nex, Ly / ney

    def compute_node_coords(i, j):
        return jnp.array([i * dx / 2, j * dy / 2])

    node_coords = jax.vmap(jax.vmap(compute_node_coords, in_axes=(0, None)), 
                           in_axes=(None, 0))(jnp.arange(nnx), jnp.arange(nny)).reshape(-1, 2)

    def compute_nconn(ie, je):
        n0 = (2 * ie)     + (2 * je)     * nnx
        n1 = (2 * ie + 1) + (2 * je)     * nnx
        n2 = (2 * ie + 2) % nnx + (2 * je)     * nnx  # wrap around in x
        n3 = (2 * ie + 2) % nnx + (2 * je + 1) * nnx
        n4 = (2 * ie + 2) % nnx + (2 * je + 2) % nny * nnx  # wrap around in x and y
        n5 = (2 * ie + 1) + (2 * je + 2) % nny * nnx
        n6 = (2 * ie)     + (2 * je + 2) % nny * nnx
        n7 = (2 * ie)     + (2 * je + 1) * nnx
        n8 = (2 * ie + 1) + (2 * je + 1) * nnx

        return jnp.array([n0, n1, n2, n3, n4, n5, n6, n7, n8], dtype=int)

    elem_indices = jnp.arange(nex * ney)
    ie_indices = elem_indices % nex  # element index in x-direction
    je_indices = elem_indices // nex  # element index in y-direction

    nconn = jax.vmap(compute_nconn)(ie_indices, je_indices)

    surfnodes = None
    bconns = None
    nvecs = None

    return node_coords, nconn, surfnodes, bconns, nvecs

