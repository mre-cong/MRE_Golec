import numpy as np
def get_wca_force(r,sigma):
    eps_constant = 100
    sigma_over_separation = sigma/r
    # potential = 4*eps_constant*(pow(sigma_over_separation,12) - pow(sigma_over_separation,6))
    force_mag = 4*eps_constant*(12*(sigma_over_separation**13)/sigma - 6*(sigma_over_separation**7)/sigma)
    return force_mag

def main():
    eq_length = 10
    sigma = 0.1*eq_length
    cutoff_length = (2**(1/6))*sigma
    ri = np.array([0.,0.,0.])
    rj = np.array([0.,-1.0/np.sqrt(2),1.0/np.sqrt(2)])
    rij = ri - rj
    rij_mag = np.sqrt(np.dot(rij,rij))
    spring_force = np.zeros((3,))
    if rij_mag <= cutoff_length:#if the spring has shrunk to 2^(1/6)*10% or less of it's equilibrium length, we want to introduce an additional repulsive force to prevent volume collapse/inversion of the volume elements
        wca_mag = get_wca_force(rij_mag,sigma)
        for i in range(3):
            spring_force[i] += wca_mag * rij[i] / rij_mag
    print(f'spring force is {spring_force}')
    spring_force_mag = np.sqrt(np.dot(spring_force,spring_force))
    print(f'spring force magnitude is {spring_force_mag}')

if __name__ == '__main__':
    main()