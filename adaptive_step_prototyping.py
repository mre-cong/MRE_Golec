import rkf45
import numpy as np

#practicing using the adaptive heun-euler method with arbitrary function f(t,y) to solve the ODE dy/dt = f(t,y). 
# first test, using f(t,y) = y + t * t
def my_func(t,y):
    f = y + t * t
    return f

def main():
    y_0 = np.ones((2,1))
    t_0 = 0
    h = 1e-3
    N_iter = 100
    y = y_0
    y_array = np.zeros((2,N_iter+1))
    t_array = np.zeros((N_iter+1,1))
    y_array[0,:] = y_0
    t_array[0] = t_0
    for i in range(N_iter):
        y, t = rkf45.heun_euler_2(y,t,my_func,h)
        y_array[i+1]
    return 0

if __name__ == "__main__":
    main()