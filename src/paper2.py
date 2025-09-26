import mujoco
import mujoco.viewer
import numpy as np
import matplotlib.pyplot as plt
import time
from pathlib import Path

# Load model and data
HERE = Path(__file__).resolve().parent             # .../rcm_controller/src
MODEL_XML = HERE.parent / "models" / "panda1.xml"   

model = mujoco.MjModel.from_xml_path(str(MODEL_XML))
data = mujoco.MjData(model)

# Simulation parameters
nq, nv, nu = model.nq, model.nv, model.nu
dt = model.opt.timestep

# Sites and IDs
ee_site          = model.site('ee_site').id
ptrocar_site     = model.site('rcm_site').id
shaft_base       = model.site('shaft_base').id   # Pi
shaft_tip        = model.site('shaft_tip').id    # Pi+1

## >>> control via joint-position servos (first 7 joints = arm)
arm_joint_names = [f"joint{i}" for i in range(1,8)]
arm_idx = np.array([model.joint(n).qposadr for n in arm_joint_names], dtype=int).ravel()

# 7 arm actuators that drive those joints (by name)
act_names = [f"actuator{i}" for i in range(1,8)]
act_idx  = np.array([model.actuator(n).id for n in act_names], dtype=int)

q0   = np.array([0.0, 0.0, 0.0, -1.0, 0.0, 1.0, -0.5, 0.004, 0.004])


#FK at q0
data.qpos[:nq] = q0
mujoco.mj_forward(model, data)

p_i    = data.site_xpos[shaft_base]
p_ip1  = data.site_xpos[shaft_tip]
p_trc  = data.site_xpos[ptrocar_site]

d = p_ip1 - p_i
d_norm = np.linalg.norm(d)
eps=1e-12
if d_norm < eps:
    # degenerate: shaft points coincide
    d_norm = eps
l_vector = d / d_norm    

d_inside = 0.09                                 # 3 cm inside the trocar
p_center = p_trc + d_inside * l_vector

# orthonormal basis {u,v} in plane ⟂ l_vector
tmp = np.array([1.0,0.0,0.0]) if abs(l_vector[0]) < 0.9 else np.array([0.0,1.0,0.0])
u = tmp - l_vector*np.dot(tmp, l_vector); u /= np.linalg.norm(u)
v = np.cross(l_vector, u)

R     = 0.01                                     # 1 cm circle
omega = 2*np.pi*0.5                              # 0.5 Hz
Ksteps = 1500                                    

# Desired position at t = 0 (theta = 0) for current circle:
p_des0 = p_center + R * u       # since cos(0)=1, sin(0)=0
#print("p_des0:", p_des0)

print("Desired start point    :", p_des0)

# DLS IK to place EE at p_des0 (small, stable)
data.qpos[:nq] = q0
mujoco.mj_forward(model, data)
for _ in range(40):
    p_ee = data.site_xpos[ee_site]
    e    = p_des0 - p_ee
    if np.linalg.norm(e) < 1e-6:
        break
    Jt = np.zeros((3, nq))
    mujoco.mj_jacSite(model, data, Jt, None, ee_site)
    lam_dls = 1e-3
    dq = Jt.T @ np.linalg.solve(Jt @ Jt.T + lam_dls*np.eye(3), e)
    q0 += dq
    data.qpos[:nq] = q0
    mujoco.mj_forward(model, data)

print("EE after  IK warm-start:", data.site_xpos[ee_site])
#Read q configuration after IK
q0 = data.qpos.copy()

p_i    = data.site_xpos[shaft_base]
p_ip1  = data.site_xpos[shaft_tip]
p_trc  = data.site_xpos[ptrocar_site]
d      = p_ip1 - p_i
eps   = 1e-12

l_vector  = d / (np.linalg.norm(d) + 1e-12)   # unit vector along the shaft (P_i -> P_{i+1})

# orthonormal basis {u,v} in plane ⟂ zhat
tmp = np.array([1.0,0.0,0.0]) if abs(l_vector[0]) < 0.9 else np.array([0.0,1.0,0.0])
u = tmp - l_vector*np.dot(tmp, l_vector); u /= np.linalg.norm(u)
v = np.cross(l_vector, u)
p_center = p_trc + d_inside * l_vector

# Set simulation state from this consistent start
q   = q0.copy()

def desired_at_time(t):
    # small true circle in allowed plane, centered inside the body
    theta = omega * t
    p_des = p_center + R*(u*np.cos(theta) + v*np.sin(theta))        
    v_des = R*omega*(-u*np.sin(theta) + v*np.cos(theta))
    return p_des, v_des


def rcm_distance_task(p_i, J_i, p_ip1, J_ip1, p_trc, eps=1e-12):

    d = p_ip1 - p_i
    d_norm = np.linalg.norm(d)
    if d_norm < eps:
        # degenerate: shaft points coincide
        d_norm = eps
    l_vector = d / d_norm                                    # (3,) unit shaft vector

    #projecting (ptrocar - pi) in the link i: λli = l_vector^T (ptrocar - pi)
    lambda_i = l_vector.T @ (p_trc - p_i)                    

    #p_RCM(q) = pi + λli * l_vector
    p_RCM = p_i + lambda_i * l_vector                         # (3,) RCM point on the shaft   

    #Distance unit vector from Prcm to Ptrocar: D^(q) = (Ptrc - Prcm(q)) / ||Ptrc - Prcm(q)||
    D_vec = p_trc - p_RCM
    D_norm = np.linalg.norm(D_vec)
    if D_norm < eps:
        D_norm = eps
    D_hat = D_vec / D_norm                                    # (3,) unit distance vector

   
    #Distance-to-trocar task: t(q) = -|| p_T - p_RCM(q) ||
    t = -D_norm                                               # scalar task value

    #Derivative of l_vector: (1/d)(I - l_vector l_vector^T) (J_ip1 - J_i) 
    dl_vector = (1.0/d_norm) * (np.eye(3) - np.outer(l_vector, l_vector)) @ (J_ip1 - J_i)   # (3,n)
   
    #Derivative of λli: dl_vector^T (ptrocar - pi) - l_vector^T J_i  
    dlambda_i = dl_vector.T @ (p_trc - p_i) - l_vector.T @ J_i   # (n,)

    #Derivative of p_RCM: J_i + dlambda_i * l_vector + λli * dl_vector  --it should be (3,n)
    J_pRCM = J_i + np.outer(l_vector, dlambda_i) + lambda_i * dl_vector   # (3,n)

    #task RCM Jacobian: J_RCM = D_hat^T * J_{p_RCM}
    J_RCM = (D_hat.reshape(1,3) @ J_pRCM)               # (1,n)


    return t, J_RCM, p_RCM, D_hat, D_norm


# #-----------Sandoval torque - level controller----------


#-----Gains (task-space PD)------#
K1 = 500.0           #primary t(q) = -|| p_T - p_RCM(q) ||  (RCM)
D1 = 100.0

# K2 = np.diag([6500.0, 1000.0, 11000.0])      # secondary EE (3x3)
# D2 = np.diag([300.0, 90.0, 550.0])

# K2 = np.diag([15875.0, 2440.0, 26875.0])      # secondary EE (3x3)
# D2 = np.diag([750.0, 225.0, 1345.0])

K2 = np.diag([19845.0, 3050.0, 33595.0])      # secondary EE (3x3)   #best !!
D2 = np.diag([940.0, 280.0, 1680.0])

# K2 = np.diag([19845.0, 3180.0, 33595.0])      # secondary EE (3x3)   
# D2 = np.diag([940.0, 295.0, 1680.0])

# K2 = np.diag([29770.0, 4575.0, 50395.0])      # secondary EE (3x3)  no
# D2 = np.diag([1410.0, 420.0, 2520.0])

# === Logging ===
t_log, dist_log = [], []
ee_traj, des_traj = [], []
ee_err_norm, leak_arr = [], []

#-------Run Viewer loop-------#
Ksteps = 1500
with mujoco.viewer.launch_passive(model, data) as viewer:
    viewer.cam.azimuth = 150
    viewer.cam.elevation = -20
    viewer.cam.distance = 0.6
    viewer.cam.lookat[:] = [0.4, 0, 0.4]

    
    # Before the loop
    p_des_list = []
    v_des_list = []
    for k in range(Ksteps):
        t = k * dt
        t_log.append(t) 

         # desired EE pos/vel
        p_des, dot_p_des = desired_at_time(t)
        a_des = - (omega**2) * (p_des - p_center)
        p_des_list.append(p_des.copy())
        v_des_list.append(dot_p_des.copy())
        #1) FK (use current state in data) - use 7-DoF state
        mujoco.mj_forward(model, data)

        # positions
        p_ee = data.site_xpos[ee_site].copy()
        p_T   = data.site_xpos[ptrocar_site].copy()
        #Points & Jacobians for RCM task (Pi, Pi+1)
        p_i   = data.site_xpos[shaft_base].copy()   # Pi
        p_ip1 = data.site_xpos[shaft_tip].copy()    # Pi+1

        #Jacobians
        J_i_full    = np.zeros((3,nq)); mujoco.mj_jacSite(model, data, J_i_full,   None, shaft_base)
        J_ip1_full  = np.zeros((3,nq)); mujoco.mj_jacSite(model, data, J_ip1_full, None, shaft_tip)
        J_task_full = np.zeros((3, nq)); mujoco.mj_jacSite(model, data, J_task_full, None, ee_site) 

        J_i, J_ip1, J2 = J_i_full[:, arm_idx], J_ip1_full[:, arm_idx], J_task_full[:, arm_idx]
        

        #2) RCM task value & Jacobian 
        t_val, J1, p_RCM, D_hat, D_norm = rcm_distance_task(p_i, J_i, p_ip1, J_ip1, p_T)

        # Mass (7x7)
        M_full = np.zeros((nv, nv)); mujoco.mj_fullM(model, M_full, data.qM)    #mass matrix (symmetric PD)
        M_arm  = M_full[np.ix_(arm_idx, arm_idx)]                               #7x7
        Minv   = np.linalg.inv(M_arm)                                           #7x7
        Lambda2 = np.linalg.inv(J2 @ (Minv @ J2.T))                             # (3,3)

        #3)Primary Torque τ1 = J1^Τ F1 with F1 = K1(td - t) + D1(dtdot - tdot)
        t_des = 0.0
        qdot_arm = data.qvel[arm_idx].copy()
        tdot = (J1 @ qdot_arm).item()                                 # tdot = D_hat.T * (dPrcm/dq) * q_dot = Jrcm(q) * qdot
      
        F1 = K1*(t_des - t_val) + D1*(0.0 - tdot)                     # scalar
        tau1 = (J1.T * F1).reshape(7,)                                # (7,)

        v_ee = J2 @ qdot_arm                                          # shape (3,)
        
        e_pos = p_des - p_ee                                          #3,
        e_vel = dot_p_des - v_ee                                      #3,
       

        F2 = K2 @ e_pos + D2 @ e_vel + (Lambda2 @ a_des)  #3,
        tau2 = J2.T @ F2             
    
        #5)Null-space projector: N2 = I - J1.T(J1^†). and J1^† =  (J1 * (M^-1) * J1.T)^-1 *J1 * (M^-1)
        Minv_J1T = np.linalg.solve(M_arm, J1.T)     
        # Scalar denominator J1 * Minv * J1^T  (+ tiny eps)
        den       = (J1 @ Minv_J1T).item() + 1e-12                   # scalar

        # LEFT dynamically-consistent pseudoinverse (1×nv)
        J1_pinv   = (Minv_J1T.T) / den                               #(1x7)

        # Null-space projector (nv×nv)
        N2        = np.eye(7) - (J1.T @ J1_pinv)                     #(7x7)


        #check projector leakage after computing N2 and tau2:
        leak = (J1 @ np.linalg.solve(M_arm, N2 @ tau2)).item()       # should be ~ 0
        leak_arr.append(leak)
        #print("leak", leak)

        #6) Combine Torques: τ = τ1 + N2 τ2
        tau_arm = tau1 + (N2 @ tau2)                                 #(7,) 
        #mj_forward before reading qfrc_bias
        mujoco.mj_forward(model, data)

        #7) Add bias : τc = τ + C(q,qdot)q_dot + G(q)
        qfrc_bias_full = data.qfrc_bias[:nv]
        tau_c_arm = tau_arm + qfrc_bias_full[arm_idx]                #(7,)
        #print(" tau_c_arm (Nm): ", tau_c_arm)
        lo = model.actuator_ctrlrange[:len(tau_c_arm),0]
        hi = model.actuator_ctrlrange[:len(tau_c_arm),1]
        sat = np.logical_or(tau_c_arm <= lo+1e-9, tau_c_arm >= hi-1e-9)
        #print("sat count:", int(sat.sum()))
        
        #8) Apply control and step
        data.ctrl[:] = 0.0
        data.qfrc_applied[:] = 0.0
       
        data.ctrl[act_idx] = tau_c_arm
        mujoco.mj_step(model, data)
    
        
         #record for plotting
        ee_traj.append(p_ee.copy())
        des_traj.append(p_des.copy())
        ee_err_norm.append(np.linalg.norm(e_pos))
        dist_log.append(D_norm)

        #viewer render
        viewer.sync()
        time.sleep(dt)  

# 1)arrays and align lengths (robust)
t_arr   = np.asarray(t_log)
pd_arr  = np.vstack(p_des_list)         # (N,3)
p_arr   = np.vstack(ee_traj)            # (N,3)
dist_arr= np.asarray(dist_log)          # (N,)
een_arr = np.asarray(ee_err_norm)       # (N,)

# make all the same length
N = min(len(t_arr), len(pd_arr), len(p_arr), len(dist_arr), len(een_arr))
t_arr    = t_arr[:N]
pd_arr   = pd_arr[:N]
p_arr    = p_arr[:N]
dist_arr = dist_arr[:N]
een_arr  = een_arr[:N]

# 2) Choose settle point 
dt = float(model.opt.timestep)
k0 = 250                                # <-- this corresponds to ~0.5 s (dt=0.002)
i0 = max(0, min(k0, N-1))               # clamp for safety

# 3) Trim to steady portion
t_steady   = t_arr[i0:]
pd_steady  = pd_arr[i0:]
p_steady   = p_arr[i0:]
dist_steady= dist_arr[i0:]
een_steady = een_arr[i0:]
Ns         = len(t_steady)

# 4) Errors and metrics
e_xyz      = pd_steady - p_steady                     # (Ns,3)
e_norm     = np.linalg.norm(e_xyz, axis=1)            # (Ns,)

EE_RMS     = 1e3*np.sqrt(np.mean(e_norm**2))         
EE_P95  =    1e3*np.percentile(e_norm, 95)        
EE_MAX  =    1e3*np.max(e_norm)                  

RCM_RMS    = 1e3*np.sqrt(np.mean(dist_steady**2))    
RCM_P95 = 1e3*np.percentile(dist_steady, 95)   
RCM_MAX = 1e3*np.max(dist_steady)               

#projector leakage stats---
leak_mean = leak_max = None
if 'leak_arr' in locals() and len(leak_arr) == len(t_arr):
    leak_abs  = np.abs(np.asarray(leak_arr)[i0:])     
    leak_mean = float(np.mean(leak_abs))
    leak_max  = float(np.max(leak_abs))

print(f"dt (s): {dt:.6f}    steady from k={i0}  (t={t_steady[0]:.3f}s)")
print(f"TIP:  RMS={EE_RMS:5.2f} mm,  P95={EE_P95:5.2f} mm,  Max={EE_MAX:5.2f} mm")
print(f"RCM:  RMS={RCM_RMS:5.2f} mm,  P95={RCM_P95:5.2f} mm,  Max={RCM_MAX:5.2f} mm")

if leak_mean is not None:
    print(f"Leakage |ℓ|: mean={leak_mean:.2e},  max={leak_max:.2e}")
else:
    print("Leakage |ℓ|: (not logged — add leak_arr.append(leak) inside your loop)")

# 5) PLOTS

# (a) 3D trajectories, steady only
fig = plt.figure()
ax  = fig.add_subplot(111, projection='3d')
ax.plot(100*p_steady[:,0],  100*p_steady[:,1],  100*p_steady[:,2],  '-', label='EE')
ax.plot(100*pd_steady[:,0], 100*pd_steady[:,1], 100*pd_steady[:,2], '--', label='Desired')
ax.legend(); ax.set_xlabel('X'); ax.set_ylabel('Y'); ax.set_zlabel('Z')
# equal aspect so the circle looks like a circle
ax.set_box_aspect((1,1,1))
plt.title('Trajectories')
# ---- Make plot 'zoomed out' ----
ax.set_xlim([37, 41])   # widen ranges a bit 
ax.set_ylim([-2,  2])
ax.set_zlim([35, 37])
plt.show()

# (b) RCM distance vs time (steady)
plt.figure()
plt.plot(t_steady, dist_steady)
# plt.plot(t_arr, dist_arr)
plt.xlabel('time (s)'); plt.ylabel('RCM distance |P_T - P_RCM| (m)')
plt.title('RCM distance convergence')
plt.grid(True); plt.show()

# (c) EE tracking error norm vs time after(steady)
plt.figure()
plt.plot(t_steady, e_norm)
# plt.plot(t_arr, np.linalg.norm((pd_arr - p_arr), axis=1))
plt.title('EE tracking error ‖p_des - p_ee‖')
plt.xlabel('time (s)'); plt.ylabel('meters')
plt.grid(True); plt.show()
