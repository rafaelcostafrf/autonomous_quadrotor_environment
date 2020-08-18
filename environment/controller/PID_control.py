import numpy as np

class pid_control():
    def __init__(self, timestep, P_att, I_att, D_att, P_pos, I_pos, D_pos):
        self.P_att = P_att
        self.I_att = I_att
        self.D_att = D_att
        
        self.P_pos = P_pos
        self.I_pos = I_pos
        self.D_pos = D_pos
        
        
        self.ts = timestep
        self.theta_accum_error = 0 
        self.phi_accum_error = 0 
        self.psi_accum_error = 0 
        self.g = 9.82
        
    def pid_int(self, pos, vel, pos_target, vel_target, accum_error, P, I, D):
        return P*(pos_target - pos)+I*(accum_error)+D*(vel_target - vel)   
    
    def lower_control(self, phi, dphi, theta, dtheta, psi, dpsi, z, dz, phi_target, theta_target, psi_target, z_target):
        dtheta_target = 0
        dphi_target = 0
        dpsi_target = 0
        dz_target = 0
        self.theta_accum_error += theta_target-theta
        self.phi_accum_error += phi_target-phi
        self.psi_accum_error += psi_target-psi
        
        u1 = (self.pid_int(z, dz, z_target, dz_target, self.z_accum_error, self.P_pos, self.I_pos, self.D_pos)+self.g)*np.cos(phi)*np.cos(psi)
        
        u2 = self.pid_int(theta, dtheta, theta_target, dtheta_target, self.theta_accum_error, self.P_att, self.I_att, self.D_att)
        u3 = self.pid_int(phi, dphi, phi_target, dphi_target, self.phi_accum_error, self.P_att, self.I_att, self.D_att)
        
        u4 = self.pid_int(psi, dpsi, psi_target, dpsi_target, self.psi_accum_error, self.P_att, self.I_att, self.D_att)

        return u1, u2, u3, u4
    
    
    def higher_control(self, x, dx, y, dy, psi, x_target, y_target):
        dx_target = 0
        dy_target = 0        
        theta_target = (self.P_pos*(x_target-x)+self.D_pos*(dx_target-dx))*np.sin(psi)
        phi_target = (-self.P_pos*(y_target-y)+self.D_pos*(dy_target-dy))*np.sin(-psi)        
        return phi_target, theta_target
    
    def control_function(self, states, targets):
        x, dx, y, dy, z, dz, phi, theta, psi, dphi, dtheta, dpsi = tuple(states)
        x_target, y_target, z_target, psi_target = tuple(targets)
        phi_target, theta_target = self.higher_control(x, dx, y, dy, x_target, y_target)
        u1, u2, u3, u4 = self.lower_control(phi, dphi, theta, dtheta, psi, dpsi, z, dz, phi_target, theta_target, psi_target, z_target)
        
        return prop_speeds
        
        
        
a = pid_control(0.01, 1, 1, 1, 1, 1, 1)
print(a.higher_control(0, 0, 0, 0, np.pi/4, 1, 1))