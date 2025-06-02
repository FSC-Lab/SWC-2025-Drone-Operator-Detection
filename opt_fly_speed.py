import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize, rosen, rosen_der

def Getkt(CT, rho, d):
    return CT * rho * d**4 / (4 * np.pi**2)

def Getktau(CP, rho, d):
    return CP * rho * d**5 / (8 * np.pi**3)


def GetOmega(W, CD, s, rho, v_infty, kt, numofrotors):
    D = 0.5 * rho * v_infty**2 * CD * s
    totalLift = np.sqrt(D**2 + W**2)
    return np.sqrt(totalLift / (numofrotors * kt)), D

def GetRequiredTorquePerMotor(omega, ktau):
    return ktau * omega**2

def GetRequiredPowerPerMotor(omega, ktau, eta, KT, Ra, Ke):
    tau = GetRequiredTorquePerMotor(omega, ktau)
    # required current
    ia = tau / (eta * KT)
    # required voltage
    u = Ra * ia + Ke * omega
    return u * ia

def GetF(W, CD, s, rho, v, kt, numofrotors,  ktau, eta, KT, Ra, Ke):
    omega, _ = GetOmega(W, CD, s, rho, v, kt, numofrotors)
    pt = GetRequiredPowerPerMotor(omega, ktau, eta, KT, Ra, Ke)
    return pt/v

if __name__ == '__main__':
    # define parameters
    W = 2 * 9.8 # weight
    print(f"weight: {W}")
    CD = 1.0 # drag coefficient
    s = 0.2 # drone reference area
    rho = 1.23
    # torque and power coefficient of the propeller
    # https://m-selig.ae.illinois.edu/props/volume-1/propDB-volume-1.html
    # 10 x 4.7 
    CP = 0.05
    CT = 0.14
    Ra = 0.028
    eta = 0.9
    KT = 0.0048
    Ke = 0.0048
    # 10 inch
    numofrotors = 4.0
    d = 0.254
    kt = Getkt(CT, rho, d)
    ktau = Getktau(CP, rho, d)
    print(f"kt : {kt}")
    print(f"ktau : {ktau}")
    # genrate a sequence of velocities
    v_infty_sq = np.linspace(1.0, 20.0, num=100)
    f = np.zeros_like(v_infty_sq)
    omega_sq = np.zeros_like(v_infty_sq)
    J_sq = np.zeros_like(v_infty_sq)
    pitch_sq = np.zeros_like(v_infty_sq)
    power_sq = np.zeros_like(v_infty_sq)
    for i , v in enumerate(v_infty_sq):
        omega, D = GetOmega(W, CD, s, rho, v, kt, numofrotors)
        pt = GetRequiredPowerPerMotor(omega, ktau, eta, KT, Ra, Ke)
        power_sq[i] = pt
        f[i] = pt/v
        omega_sq[i] = omega
        pitch_sq[i] = np.arctan(D / W)
        J_sq[i] = v * np.sin(pitch_sq[i]) / (omega * d/(2 * np.pi))
    
    fp = lambda x: GetF(W, CD, s, rho, x, kt, numofrotors,  ktau, eta, KT, Ra, Ke)
    v_star = minimize(fp, x0=10.0)
    
    print(v_star)
    omega_star, D_star = GetOmega(W, CD, s, rho, v_star['x'], kt, numofrotors)
    pt_star = GetRequiredPowerPerMotor(omega_star, ktau, eta, KT, Ra, Ke)
    print(f"pt_star : {pt_star}")


    plt.figure(figsize=(5,5), dpi= 100)
    plt.plot(v_infty_sq, f, label='cost function vs flight speed')
    plt.plot(v_star.x, v_star.fun, '-r*', label='optimal cruise speed', linewidth=3.0)
    plt.grid(True)
    plt.xlabel("v(m/s)")
    plt.ylabel("f")
    plt.legend()
    
    plt.figure(figsize=(8,12), dpi= 100)
    plt.subplot(4, 1, 1)
    plt.plot(v_infty_sq, 57.3 * pitch_sq)
    plt.grid(True)
    plt.xlabel("v(m/s)")
    plt.ylabel("pitch(deg)")
    
    plt.subplot(4, 1, 2)
    plt.plot(v_infty_sq, J_sq)
    plt.grid(True)
    plt.xlabel("v(m/s)")
    plt.ylabel("J")
    
    plt.subplot(4, 1, 3)
    plt.plot(v_infty_sq, omega_sq)
    plt.grid(True)
    plt.xlabel("v(m/s)")
    plt.ylabel(" omega(rad/s)")

    plt.subplot(4, 1, 4)
    plt.plot(v_infty_sq, power_sq)
    plt.grid(True)
    plt.xlabel("v(m/s)")
    plt.ylabel(" power(w)")
    plt.show()