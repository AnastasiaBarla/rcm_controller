import mujoco
import mujoco.viewer
import numpy as np
import matplotlib.pyplot as plt
import time


# Load model and data
MODEL_XML = "/home/anastasia/Desktop/Diplwmatikh/MuJoCo/franka_emika_panda/panda.xml"
model = mujoco.MjModel.from_xml_path(MODEL_XML)
data = mujoco.MjData(model)

# Simulation parameters
nq = model.nq
dt = model.opt.timestep

#control via joint-position servos (first 7 joints = arm)
arm_joint_names = [f"joint{i}" for i in range(1,8)]
arm_idx = np.array([model.joint(n).qposadr for n in arm_joint_names], dtype=int).ravel()

#7 arm actuators that drive those joints (by name)
act_names = [f"actuator{i}" for i in range(1,8)]
act_idx  = np.array([model.actuator(n).id for n in act_names], dtype=int)

alpha = 1. * dt       # small position step from the velocity-like solution


# Sites and IDs
ee_site          = model.site('ee_site').id
ptrocar_site     = model.site('rcm_site').id
shaft_base       = model.site('shaft_base').id   # Pi
shaft_tip        = model.site('shaft_tip').id    # Pi+1

# Control gains
Kt = np.diag([400.0, 400.0, 700.0])   # task gain (EE)
Kr = np.eye(3) * 800.0                # RCM gain  trocar
damp = 3e-4                           # DLS damping (stable)

q0   = np.array([0.0, 0.0, 0.0, -1.0, 0.0, 1.0, -0.5, 0.004, 0.004])


#FK at q0
data.qpos[:nq] = q0
mujoco.mj_forward(model, data)

# ---------- λ0 from projection onto the segment P_i->P_{i+1}----------
p_i    = data.site_xpos[shaft_base]
p_ip1  = data.site_xpos[shaft_tip]
p_trc  = data.site_xpos[ptrocar_site]
d_seg  = p_ip1 - p_i
lam0   = np.clip(np.dot(p_trc - p_i, d_seg) / (np.dot(d_seg, d_seg) + 1e-12), 0.0, 1.0)

#State
q = q0.copy()
lam = lam0

# use current shaft direction (normalized) for plane & center = trocar + d_inside * zhat
zhat  = d_seg / (np.linalg.norm(d_seg) + 1e-12)   # unit vector along the shaft (P_i -> P_{i+1})
d_inside = 0.09                                 # 3 cm inside the trocar
p_center = p_trc + d_inside * zhat
#print("p_center:", p_center)


# orthonormal basis {u,v} in plane ⟂ zhat
tmp = np.array([1.0,0.0,0.0]) if abs(zhat[0]) < 0.9 else np.array([0.0,1.0,0.0])
u = tmp - zhat*np.dot(tmp, zhat); u /= np.linalg.norm(u)
v = np.cross(zhat, u)

R     = 0.01                                     # 1 cm circle
omega = 2*np.pi*0.5                              # speed of rotation (0.5 Hz)
Ksteps = 1500                                    # ~15 seconds of motion

# Desired position at t = 0 (theta = 0) for your current circle:
p_des0 = p_center + R * u       # since cos(0)=1, sin(0)=0
#print("p_des0:", p_des0)

print("Desired start point    :", p_des0)

# DLS IK to place EE at p_des0
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


# IMPORTANT: recompute λ0 from the new q0 (projection on the segment Pi->Pi+1)
p_i    = data.site_xpos[shaft_base]
p_ip1  = data.site_xpos[shaft_tip]
p_trc  = data.site_xpos[ptrocar_site]
d      = p_ip1 - p_i
eps   = 1e-12
lam0   = float(np.clip(np.dot(p_trc - p_i, d) / (np.dot(d, d) + eps), 0.0, 1.0))


zhat  = d / (np.linalg.norm(d) + 1e-12)   # unit vector along the shaft (P_i -> P_{i+1})
# orthonormal basis {u,v} in plane ⟂ zhat
tmp = np.array([1.0,0.0,0.0]) if abs(zhat[0]) < 0.9 else np.array([0.0,1.0,0.0])
u = tmp - zhat*np.dot(tmp, zhat); u /= (np.linalg.norm(u)+1e-12)
v = np.cross(zhat, u)
p_center = p_trc + d_inside * zhat
#print("p_center.after", p_center)

# Set simulation state from this consistent start
q   = q0.copy()


def desired_at_time(t):
    # small true circle in allowed plane, centered inside the body
    theta = omega * t
    p_des = p_center + R*(u*np.cos(theta) + v*np.sin(theta))        
    v_des = R*omega*(-u*np.sin(theta) + v*np.cos(theta))
    return p_des, v_des

# ---------- logging ----------
ee_traj, des_traj, rcm_errs,ee_err_norm, t_log = [], [], [], [], []
k_plane=500
k_along=4000
lam_hist      = []   
lamdot_hist   = []  

# Viewer setup
with mujoco.viewer.launch_passive(model, data) as viewer:
    viewer.cam.azimuth = 150
    viewer.cam.elevation = -20
    viewer.cam.distance = 0.7  
    viewer.cam.lookat[:] = [0.4, 0, 0.4]

    for k in range(Ksteps):
        t = k * dt
        t_log.append(t) 
        
         # desired EE pos/vel
        p_des, dot_p_des = desired_at_time(t)

        # use current plant state for FK/Jacobians
        mujoco.mj_forward(model, data)

        # Points
        p_i   = data.site_xpos[shaft_base]   # Pi
        p_ip1 = data.site_xpos[shaft_tip]    # Pi+1

        # Segment
        d = p_ip1 - p_i                         # Pi+1 - Pi
        dhat = d / (np.linalg.norm(d) + 1e-12)  # unit shaft direction
        

        # Variable RCM point (on the segment)
        p_rcm_var = p_i + lam * d

        # Position Jacobians (pos only) of Pi and Pi+1
        J_i_full    = np.zeros((3,nq)); mujoco.mj_jacSite(model, data, J_i_full,   None, shaft_base)
        J_ip1_full  = np.zeros((3,nq)); mujoco.mj_jacSite(model, data, J_ip1_full, None, shaft_tip)
        J_task_full = np.zeros((3, nq)); mujoco.mj_jacSite(model, data, J_task_full, None, ee_site)   # 3×nq
        
        # ---- keep ONLY the 7 arm columns in the solver ----
        J_i   = J_i_full[:,   arm_idx]                    # (3 × 7)
        J_ip1 = J_ip1_full[:, arm_idx]                    # (3 × 7)
        J_task= J_task_full[:,arm_idx].squeeze()          # (3 × 7)
        
        # Extended Jacobian
        # Paper J_RCM(q,λ) = [ J_i + λ(J_{i+1}-J_i) , (Pi+1 - Pi) ]
        J_rcm_q   = (J_i + lam * (J_ip1 - J_i)).squeeze()  # (3 x nq)
        J_rcm_lam = d.reshape(3,1)             # (3 x 1)
    
        
        top = np.hstack([J_task,    np.zeros((3,1))])      # EE pos task 
        bot = np.hstack([J_rcm_q,   J_rcm_lam])            # paper RCM block
        J_ext = np.vstack([top, bot])
       
        #errors
        p_ee = data.site_xpos[ee_site].copy()
        e_pos = (p_des - p_ee)                           

        #each step, after you compute zhat (= unit shaft direction)
        P_plane = np.eye(3) - np.outer(zhat, zhat)  # projector onto plane ⟂ shaft
        Kt = k_plane * P_plane + k_along * np.outer(zhat, zhat)
       # print("Kt", Kt)

        # RHS: drive P_RCM to P_trocar
        e_rcm = (p_trc - p_rcm_var)                          #error vector
        b_pos = Kt @ e_pos + dot_p_des
        b_rcm = Kr @ e_rcm
        b_ext = np.concatenate([b_pos, b_rcm])


        #Damped Least-Squares solve J_ext xdot = b_ext
        m = J_ext.shape[0]                             
        A    = J_ext @ J_ext.T + damp*np.eye(m)         
        Ainv = np.linalg.solve(A, np.eye(m))                 # this is A^{-1} (via solves)
        Jsharp = J_ext.T @ Ainv                              # right pseudoinverse

        xdot_ext = Jsharp @ b_ext                            # minimum‑norm solution

        #add a null-space term to bias λ toward λ0 - SVD
        U, S, Vt = np.linalg.svd(J_ext, full_matrices=True)
       
        # numerical rank
        tol   = 1e-10 * S[0] if S.size else 1e-10
        rank  = int(np.sum(S > tol))
        
        n     = J_ext.shape[1]  
        nullity = n - rank
        Z = Vt[rank:, :].T            # (nq+1) x nullity  — columns span the null-space
        
        # Orthonormalize those columns for numerical robustness
        Z_orth, _ = np.linalg.qr(Z)            # shape (n, nullity)

        # ----- pick a null direction that prefers increasing λ -----
        e_lambda = np.zeros(J_ext.shape[1]); e_lambda[-1] = 1.0

        # Project "I want λ" into the null space: v = Z_orth (Z_orth^T e_lambda)
        vn = Z_orth @ (Z_orth.T @ e_lambda)

        # If numerically near zero (degenerate), just take the biggest-λ column
        if np.linalg.norm(vn) < 1e-12:
            idx = np.argmax(np.abs(Z_orth[-1, :]))
            vn = Z_orth[:, idx].copy()

        # Make it push λ toward (lam0 - lam)
        if np.sign(vn[-1]) != np.sign(lam0 - lam):
            vn = -vn

        # Normalize (optional)
        vn /= (np.linalg.norm(vn) + 1e-12)
    #     print("vn", vn)
    #     print("J_ext.shape", J_ext.shape,
    #   "  nullity =", Z_orth.shape[1],
    #   "  ||J_ext vn|| =", np.linalg.norm(J_ext @ vn),
    #   "  v[-1] (λ component) =", vn[-1])

        # Null-space push to bias λ toward λ0, projected so it won't affect the task
        k_lambda = 1.0  
        delta    = (lam0 - lam)
        w_ext    = k_lambda * delta * vn

        N = np.eye(J_ext.shape[1]) - Jsharp @ J_ext      
        ns_term  = N @ w_ext
        #print(f"||N w|| = {np.linalg.norm(ns_term):.3e}")

        xdot_ext += ns_term

        ratio = np.linalg.norm(ns_term) / (np.linalg.norm(xdot_ext) + 1e-12)
        # e.g., only apply if it’s at least 5–10% of the main update
        if ratio > 0.1:
            xdot_ext += ns_term

        #convert velocity-like solution to joint position target and send to servos
        dq      = xdot_ext[:len(arm_idx)]                    # (7,)
        lam_dot = float(xdot_ext[-1])
        lamdot_hist.append(lam_dot) 

        q_des = data.qpos.copy()
      
        q_des[arm_idx] += alpha * dq                         # step only the 7 arm joints
      
        # push targets to the MuJoCo PD servos
        data.ctrl[:] = 0.0                                   # clear any old commands
        
        data.ctrl[act_idx] = q_des[arm_idx]#.flatten()       # position targets for MuJoCo PD
        mujoco.mj_step(model, data)                          # advance the simulator

  
        lam = float(np.clip(lam + lam_dot*dt, 0.0, 1.0))
        lam_hist.append(lam) 

        #record for plotting
        ee_traj.append(p_ee.copy())
        des_traj.append(p_des.copy())
        rcm_errs.append(np.linalg.norm(np.cross(e_rcm, dhat)))  
       
        viewer.sync()
        time.sleep(dt)
        

# Plot trajectories
ee_traj  = np.array(ee_traj)
des_traj = np.array(des_traj)
rcm_errs = np.array(rcm_errs)
ee_err_norm = np.array(ee_err_norm)
t_arr    = np.array(t_log)
lamdot_hist = np.array(lamdot_hist)
lam_hist    = np.array(lam_hist)

#settle time 
T_settle = 0.5
i0 = np.searchsorted(t_arr, T_settle)        # first index where t >= T_settle
mask = t_arr >= T_settle                     # boolean mask (equivalent)


# ---- tip tracking ----
ee_err = np.linalg.norm(des_traj - ee_traj, axis=1)
ee_rms = np.sqrt(np.mean(ee_err[mask]**2))
ee_p95 = np.percentile(ee_err[mask], 95)
ee_max = np.max(ee_err[mask])

# ---- RCM constraint ----
rcm_rms = np.sqrt(np.mean(rcm_errs[mask]**2))
rcm_p95    = np.percentile(rcm_errs[mask], 95)
rcm_max    = np.max(rcm_errs[mask])

# radius error (deviation from R)
radius_err = np.abs(np.linalg.norm(ee_traj[mask] - p_center, axis=1) - R)
rad_mean   = np.mean(radius_err)
rad_p95    = np.percentile(radius_err, 95)

lam_mean   = float(np.mean(lam_hist[mask]))
lam_std    = float(np.std(lam_hist[mask]))
lam_min    = float(np.min(lam_hist[mask]))
lam_max    = float(np.max(lam_hist[mask]))

print(f"N post-settle: {len(ee_err[mask])}  (settle at t={T_settle:.3f}s)")
print(f"TIP:  RMS error: {ee_rms*1e3:5.2f} mm,  P95={ee_p95*1e3:5.2f} mm,  Max={ee_max*1e3:5.2f} mm")
print(f"RCM:  RMS error: {rcm_rms*1e3:5.2f} mm,  P95={rcm_p95*1e3:5.2f} mm,  Max={rcm_max*1e3:5.2f} mm")
print(f"      Radius mean err={rad_mean*1e3:5.2f} mm (P95={rad_p95*1e3:5.2f} mm)")
print(f"λ:    mean={lam_mean:.3f}, std={lam_std:.3f}, range=[{lam_min:.3f}, {lam_max:.3f}]")


fig = plt.figure()
ax  = fig.add_subplot(111, projection='3d')

# ax.plot  (ee_traj[:,0], ee_traj[:,1], ee_traj[:,2], label="EE")
# ax.plot  (des_traj[:,0],des_traj[:,1],des_traj[:,2],'--',label="Desired")
ax.plot  (ee_traj[mask,0], ee_traj[mask,1], ee_traj[mask,2], label="EE")
ax.plot  (des_traj[mask,0],des_traj[mask,1],des_traj[mask,2],'--',label="Desired")
ax.legend()
ax.set_xlabel("X"); ax.set_ylabel("Y"); ax.set_zlabel("Z")
plt.show()

#----------RCM error plot---------
plt.figure()
plt.plot(t_arr[mask], rcm_errs[mask])
#plt.plot(t_log, rcm_errs)
plt.title("RCM (distance-to-line) monitor")
plt.xlabel("time (s)"); plt.ylabel("||e_rcm||")
plt.grid(True)
plt.show()

#--------- EE error plot---------
plt.figure()
plt.plot(t_arr[mask], ee_err[mask])
#plt.plot(t_log, ee_err_norm)
plt.title("EE tracking error ||p_des - p_ee||")
plt.xlabel("time (s)")
plt.ylabel("meters")
plt.grid(True)
plt.show()

#---------λ(t)----------
plt.figure()
plt.plot(t_arr[mask], lam_hist[mask])
#plt.plot(t_log, ee_err_norm)
plt.title("Insertion parameter λ(t)")
plt.xlabel("time (s)")
plt.ylabel("λ")
plt.grid(True)
plt.show()

#---------λdot(t)----------
plt.figure()
plt.plot(t_arr[mask], lamdot_hist[mask])
plt.axhline(0.0, lw=0.8)            # zero reference
#plt.plot(t_log, ee_err_norm)
plt.title("Insertion rate λ(t)")
plt.xlabel("time (s)")
plt.ylabel("$\\dot{\\lambda}$ (1/s)")
plt.grid(True)
plt.show()