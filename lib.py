import sys
import numpy as np
from functools import lru_cache
from numba import jit
from potential_minnesota import get_minnesota_me
#from potential_v18 import get_v18_me

# GLOBAL PARAMS
nn_potential = "minnesota" # in "minnesota", "v18"
NUM_PARTICLES = 14 # number of (magic) particles 
DENSITY = 0.36 # matter density

NMAX = 2 # max grid size [-NMAX, NMAX]^3
KMAX = 1 # max momentum

LATTICE_SIZE = (NUM_PARTICLES/DENSITY)**(1.0/3.0)
dk = 2.0 * np.pi / LATTICE_SIZE
SPIN_DEGENERACY = 2

########### UTILS #############

@lru_cache(maxsize = None)
@jit(nopython=True)
def spin_at_index(i):
    return ( 1 - 2 * np.remainder(i, 2) ) 

########### /UTILS ############


def generate_grid_momenta_and_basis():
    """
    3d lattice :
      -Nmax . . . . . . . Nmax
      .                   . .
     .                   .  .
    .                   .   .
    -Nmax . . . . . Nmax    . 
    .                .      .
    .                .     .
    .                .    .
    .                .   .
    -Nmax . . . . . Nmax 
    """
    grid = []
    n_interval = range(-NMAX, NMAX+1)
    for i in n_interval:
        for j in n_interval:
            for k in n_interval:
                grid.append(tuple([i,j,k]))

    #order by norm
    norm=np.zeros(len(grid),dtype=int)
    for i, vec in enumerate(grid):
        npvec=np.array(vec,dtype=int)
        norm[i]=np.dot(npvec,npvec)
    index=np.argsort(norm)
    grid1 = []
    for i, ind in enumerate(index):
        grid1.append(grid[ind])
    grid = grid1

    momenta = [tuple(j * dk for j in point) for point in grid]
    basis = [[point, point] for point in grid]
    basis = [p for pp in basis for p in pp]
    return [grid, momenta, basis]

def calc_kinetic_energy(basis):
    nucleon_mass = 938.92
    hbarc = 197.33
    E = 0.
    for i in range(NUM_PARTICLES):
        point = basis[i]
        p = np.array(point)*dk
        p2 = np.dot(p, p)
        E += 0.5*hbarc**2*p2/nucleon_mass
    return E
        
def calc_HF_minn(basis):
    E = calc_kinetic_energy(basis)
    pot = 0.0
    for i in range(NUM_PARTICLES):
        m_i = tuple( [p*dk for p in basis[i]] )
        si = spin_at_index(i) #1 - 2 * np.remainder(i, 2) # spin at index i
        for j in range(NUM_PARTICLES):
            m_j = tuple( [p*dk for p in basis[j]] )
            sj = spin_at_index(j) #1 - 2 * np.remainder(j, 2) # spin at index j
            p_rel = tuple([0.5*(e - e1) for e,e1 in zip(m_i, m_j)]) #0.5 * (m_i - m_j)
            p_rel_neg = tuple([-e for e in p_rel])
            pot += 0.5 * (  get_minnesota_me(p_rel,si,sj, p_rel,si,sj, LATTICE_SIZE)
                           - get_minnesota_me(p_rel,si,sj,p_rel_neg,sj,si, LATTICE_SIZE) )
    E += pot
    return E 

def calc_two_particle_me(channels_in, channels_out, rel_momenta_in, rel_momenta_out):
    """
    return list of matrices V_iioo for each channel
    """
    pot = []
    for chan_iter, chan_in in enumerate(channels_in):
        num_two_particle_states_in = len(rel_momenta_in[chan_iter])
        num_two_particle_states_out= len(rel_momenta_out[chan_iter])
        matrix = np.zeros( (num_two_particle_states_out, num_two_particle_states_in) )
        for i, tmp_in in enumerate(rel_momenta_in[chan_iter]):
            [rel_p_in, [s1_in, s2_in]] = tmp_in
            rel_p_in = tuple(rel_p_in * dk * 0.5)
            for j, tmp_out in enumerate(rel_momenta_out[chan_iter]):        
                [rel_p_out, [s1_out, s2_out]] = tmp_out                
                rel_p_out = tuple(rel_p_out * dk * 0.5)
                rel_p_out_neg = tuple([-e for e in rel_p_out])
                matrix[j, i] = (get_minnesota_me( rel_p_out, s1_out, s2_out, rel_p_in, s1_in, s2_in, LATTICE_SIZE )
                                - get_minnesota_me(rel_p_out_neg, s2_out, s1_out, rel_p_in, s1_in, s2_in, LATTICE_SIZE) )
                if num_two_particle_states_in == num_two_particle_states_out:
                    matrix[i,j] = matrix[j,i]
        pot.append(matrix)
    return pot

def generate_channels(basis):
    """
    returns p_i + p_j for i,j in hole states and i,j in particle states
    """
    channels_hh,channels_pp, rel_momenta_hh, rel_momenta_pp = [], [], [], []
    for i, p1 in enumerate(basis[0:NUM_PARTICLES]):
        p1 = np.array(p1)
        spin1 = spin_at_index(i)
        for j, p2 in enumerate(basis[0:NUM_PARTICLES]):
            if (i == j): continue
            p2 = np.array(p2)
            spin2 = spin_at_index(j)
            p12 = p1 + p2
            def check_existing():
                for channel in channels_hh:
                    if (channel == p12).all():
                        return True
                return False
            prel = p1 - p2
            spins = np.array([spin1,spin2],dtype=int)
            ps = [prel, spins]
            if check_existing():
                # the channel already exists 
                # find the index of that channel in channels_hh
                indices = [i for i, x in enumerate(channels_hh) if (x == p12).all()]
                if len(indices) > 1:
                    print("ERROR: multiple indices of channels_hh") 
                    sys.exit()
                idx_chan = indices[0]
                # append relative momentum to the list rel_momenta_pp
                rel_momenta_hh[idx_chan].append(ps)
            else:
                # it is a new hh channel
                channels_hh.append( p12 )
                rel_momenta_hh.append ([ ps ])

    min_norm_hh = min([np.linalg.norm(x) for x in channels_hh])
    max_norm_hh = max([np.dot(x, x) for x in channels_hh])
    
    size_basis = len(basis)
    for i, p1 in enumerate(basis[NUM_PARTICLES: size_basis]):
        p1 = np.array(p1)
        spin1 = spin_at_index(i)
        for j, p2 in enumerate(basis[NUM_PARTICLES: size_basis]):
            p2 = np.array(p2)
            spin2 = spin_at_index(j)
            p12 = p1 + p2
            p12_norm = np.dot(p12, p12)
            if p12_norm > max_norm_hh:
                continue
            def check_existing():
                for channel in channels_pp:
                    if (channel == p12).all():
                        return True
                return False
            prel = p1 - p2
            spins = np.array([spin1,spin2],dtype=int)
            ps = [prel, spins]
            if check_existing():
                # the channel already exists 
                # find the index of that channel in channels_pp
                indices = [i for i, x in enumerate(channels_pp) if (x == p12).all()]
                if len(indices) > 1:
                    print("ERROR: multiple indices of channels_pp") 
                    sys.exit()
                idx_chan = indices[0]
                # append relative momentum to the list rel_momenta_pp
                rel_momenta_pp[idx_chan].append(ps)
            else:
                # it is a new pp channel which should be existing in channels_hh as well 
                for chan_hh in channels_hh:
                    if (chan_hh == p12).all():
                        channels_pp.append( p12 )
                        rel_momenta_pp.append ([ ps ])
                        break

    ordered_channels_pp, ordered_rel_momenta_pp = [], []
    for i, chan_hh in enumerate(channels_hh):
        for j, chan_pp in enumerate(channels_pp):
            if (chan_pp==chan_hh).all():
                ordered_channels_pp.append(chan_pp)
                ordered_rel_momenta_pp.append(rel_momenta_pp[j])
    channels_pp = ordered_channels_pp
    rel_momenta_pp = ordered_rel_momenta_pp
    return channels_hh, channels_pp, rel_momenta_hh, rel_momenta_pp

def get_nn_matrix_elements():
    """
    < | V | >
    """
    if nn_potential == "minnesota":
        return get_minnesotta_me()
    if nn_potential == "v18":
        return get_v18_me()
    return 0

def solve_ccd():
    """
    """
    return 0

if __name__ == "__main__":
    nvec, kvec, basis = generate_grid_momenta_and_basis()
    E_kin = calc_kinetic_energy(basis)
    print("E_kin = ", E_kin)
    print("E_kin/N = ", E_kin/NUM_PARTICLES)
    E_HF = calc_HF_minn(basis)
    print("E_HF = ", E_HF)

    channels_hh, channels_pp, rel_momenta_hh, rel_momenta_pp = generate_channels(basis)

    print ("number of hh channels is " + str(len(channels_hh)) + "; number of pp channels is " + str(len(channels_pp)) )
    if (len(channels_hh) != len(channels_pp)):
        print("ERROR: number of hh and pp channels is not equal")
        sys.exit()

    count = 0
    for i, channel in enumerate(channels_hh):
        count += len(rel_momenta_hh[i])
    print('number of contributing hh states', count)

    count = 0
    for i, channel in enumerate(channels_pp):
        count += len(rel_momenta_pp[i])
    print('number of contributing pp states', count)
    Vhhhh = calc_two_particle_me(channels_hh, channels_hh, rel_momenta_hh, rel_momenta_hh)
    Vpppp = calc_two_particle_me(channels_pp, channels_pp, rel_momenta_pp, rel_momenta_pp)
    Vpphh = calc_two_particle_me(channels_hh, channels_pp, rel_momenta_hh, rel_momenta_pp)

