import numpy as np
from numba import jit
from functools import lru_cache
@lru_cache(maxsize = None)
@jit(nopython=True)
def get_minnesota_me(p_out,s1_out,s2_out,p_in,s1_in,s2_in,L):
    """
    s - python numbers
    p - python tuples
    """
    delta_s1_in_s1_out = (s1_in * s1_out + 1)/2.
    delta_s2_in_s2_out = (s2_in * s2_out + 1)/2.
    delta_s1_in_s2_out = (s1_in * s2_out + 1)/2.
    delta_s2_in_s1_out = (s2_in * s1_out + 1)/2.

    spin_deltas = 0.5 * ( delta_s1_in_s1_out * delta_s2_in_s2_out -
                          delta_s1_in_s2_out * delta_s2_in_s1_out )

    if spin_deltas == 0.:
        return 0
    
    Vr = 200.0
    Vs = -91.85
    kappaR = 1.487
    kappaS = 0.465

    p_in = np.array(p_in)
    p_out = np.array(p_out)
    q = p_out - p_in
    q2 = np.dot(q,q)

    me = spin_deltas * (  Vr*np.exp(-0.25*q2/kappaR) / (L*np.sqrt(kappaR))**3 + + Vs*np.exp(-0.25*q2/kappaS) / (L*np.sqrt(kappaS))**3 )

    me *= (np.sqrt(np.pi))**3

    return me

@jit
def spin2spinor(s):
    """
    Makes a two-component spinor of an integer s
    param s: spin = +/- 1
    return: two-component numpy array [1,0] for up and [0,1] for down
    """
    a =  1 << int( (1 + s) * 0.5 ) 
    return np.array( [ a /2, a % 2 ] ) 

@jit(nopython=True)
def minnesota_nn(p_out,s1_out,s2_out,p_in,s1_in,s2_in,Lbox):
    """
    The Minnesota potential between two neutrons, not yet anti-symmetrized 
    param p_out: relative out momentum
    param p_in : relative in momentum
    param s1_out, s2_out: spin projections of out particles 1 and 2
    param s1_in, s2_in  : spin projections of in particles 1 and 2
    Lbox : size of momentum box
    return: value of potential in MeV; not anti-symmetrized!
    """
    # parameters. VT is not active between two neutrons (no triplet)
    VR = 200.0
    VS = -91.85  # sign typo in Lecture Notes Physics 936, Chap. 8
    kappaR = 1.487
    kappaS = 0.465

    #print(Lbox)
    
    qvec = p_out-p_in
    q2 = np.dot(qvec,qvec)
    qvec = p_out-p_in
    q2 = np.dot(qvec,qvec)
    
    s1_i = spin2spinor(s1_in) 
    s2_i = spin2spinor(s2_in)
    s1_o = spin2spinor(s1_out)
    s2_o = spin2spinor(s2_out)
    
    spin_part = 0.5 * ( np.dot(s1_i,s1_o) * np.dot(s2_i,s2_o)
                      - np.dot(s1_i,s2_o) * np.dot(s2_i,s1_o) )
    
    pot = spin_part * (  VR*np.exp(-0.25*q2/kappaR) / (Lbox*np.sqrt(kappaR))**3 
                       + VS*np.exp(-0.25*q2/kappaS) / (Lbox*np.sqrt(kappaS))**3 )
    
    pot *= (np.sqrt(np.pi))**3 

    # if spin_part != 0:
    #     if s1_out < 1:
    #         print( np.dot(s1_i,s1_o),  np.dot(s2_i,s2_o), np.dot(s1_i,s1_o) * np.dot(s2_i,s2_o) )
    #         print(p_out,s1_out,s2_out,p_in,s1_in,s2_in,Lbox)
    #         print(spin_part, pot)
    #         sys.exit()
    return pot



if __name__ == "__main__":
    # for i in range(4):
    #     for s1_out in [-1, 1]:
    #         for s1_in in [-1, 1]:
    #             for s2_out in [-1, 1]:
    #                 for s2_in in [-1, 1]:
    #                     minnesota_new = get_minnesota_me(tuple([i%3, (i+1)%3., (i+2)%3]), s1_out, s2_out, tuple([(i+2)%3., (i+3)%3, i%3]), s1_in, s2_in, 3.815714141844439 )
                                        
    #                     minnesota_old = minnesota_nn(np.array([i%3, (i+1)%3., (i+2)%3]), s1_out, s2_out, np.array([(i+2)%3., (i+3)%3, i%3]), s1_in, s2_in, 3.815714141844439 )

    #                     print (minnesota_new, minnesota_old)
                                         
                        
    for i in range(1000000):
        #get_minnesota_me(tuple(np.array([0., 0., 0.])), 1, -1, tuple(np.array([0., 0., 0.])), 1, -1, 3.815714141844439 )
        get_minnesota_me(tuple([0., 0., 0.]), 1, -1, tuple([0., 0., 0.]), 1, -1, 3.815714141844439 )
    #     minnesota_nn(np.array([0., 0., 0.]), 1, -1, np.array([0., 0., 0.]), 1, -1, 3.815714141844439 )
    #     minnesota_nn(np.array([0., 0., 0.]), -1, 1, np.array([0., 0., 0.]), -1, 1, 3.815714141844439 )
