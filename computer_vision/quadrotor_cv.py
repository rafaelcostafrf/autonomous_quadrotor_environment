import cv2 as cv
import numpy as np
import time
from collections import deque
from scipy.spatial.transform import Rotation as R

class computer_vision():
    def __init__(self, render, quad_model, quad_env, quad_sens, quad_pos, img_buffer, camera_cal, mydir, IMG_POS_DETER):
        self.mtx = camera_cal.mtx
        self.dist = camera_cal.dist
        self.IMG_POS_DETER = IMG_POS_DETER
        self.mydir = mydir
        self.quad_model = quad_model
        self.quad_env = quad_env
        self.quad_sens = quad_sens
        self.quad_pos = quad_pos
        self.image_pos = None
        self.vel_sens = deque(maxlen=100)
        self.vel_img = deque(maxlen=100)
        self.render = render  
        
        self.img_buffer = img_buffer
        
        self.fast = cv.FastFeatureDetector_create()
        self.fast.setThreshold(20)
        
        self.criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        self.render.taskMgr.add(self.pos_deter, 'Position Determination')
        
        self.render.cam_1.setPos(0, 0, 0.01)
        self.render.cam_1.lookAt(0, 0, 0)
        self.render.quad_model.setPos(0, 0, 0)
        self.render.quad_model.setHpr(0, 0, 0)
        self.render.cam_1.reparentTo(self.render.quad_model)

        self.criteria = (cv.TERM_CRITERIA_EPS + cv.TermCriteria_COUNT, 1, 0.0001) 
        self.nCornersCols = 9
        self.nCornersRows = 6
        self.objp = np.zeros((self.nCornersCols*self.nCornersRows, 3), np.float32)
        self.checker_scale = render.checker_scale
        self.checker_sqr_size = render.checker_sqr_size
        self.objp[:,:2] = (np.mgrid[0:self.nCornersCols, 0:self.nCornersRows].T.reshape(-1,2))*self.checker_scale*self.checker_sqr_size

        
    def draw(self, img, corners, imgpts):
        corner = tuple(corners[0].ravel())
        img = cv.line(img, corner, tuple(imgpts[0].ravel()), (255, 0, 0), 1)
        img = cv.line(img, corner, tuple(imgpts[1].ravel()), (0, 255, 0), 1)
        img = cv.line(img, corner, tuple(imgpts[2].ravel()), (0, 0, 255), 1)
        return img    
    
    def pos_deter(self, task):
        if self.quad_env.done or task.frame == 0:
            self.time_total_img = []
            self.image_pos = None
            self.vel_sens = deque(maxlen=100)
            self.vel_img = deque(maxlen=100)
        if self.IMG_POS_DETER:
            time_iter = time.time()
            if task.frame % 10 == 0:           
                ret, image = self.img_buffer.get_image()
                if ret:
                    img = cv.cvtColor(image, cv.COLOR_RGBA2BGR)
                    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
                    fast_gray = cv.resize(gray, None, fx=1, fy=1)
                    corner_good = self.fast.detect(fast_gray)
                    if len(corner_good) > 50:
                        ret, corners = cv.findChessboardCorners(img, (self.nCornersCols, self.nCornersRows),
                                                                cv.CALIB_CB_ADAPTIVE_THRESH + cv.CALIB_CB_NORMALIZE_IMAGE + cv.CALIB_CB_FILTER_QUADS+ cv.CALIB_CB_FAST_CHECK)
                        if ret:
                            # corners = cv.cornerSubPix(gray,corners,(11,11),(-1,-1),self.criteria)
                            ret, rvecs, tvecs = cv.solvePnP(self.objp, corners, self.mtx, self.dist)
                            if ret:
    
                                axis = np.float32([[3,0,0], [0,3,0], [0,0,-3]]).reshape(-1,3)
                                imgpts, jac = cv.projectPoints(axis, rvecs, tvecs, self.mtx, self.dist)
                                imgpts = imgpts.astype(np.int)
        
                                real_state = np.concatenate((self.quad_env.state[0:5:2], self.quad_env.state[6:10]))
                                # rvecs[2] *= -1 
                                r = R.from_rotvec(rvecs.flatten()).inv() 
                                trans = np.dot(r.as_matrix(), tvecs).flatten() 
                                trans[0] *= -1
                                trans[1] *= -1
                                trans[2] += -5.01
                                
                                euler = r.as_euler('xyz')
                                euler[0:2] *= -1
                                r = R.from_euler('xyz', euler)
                                quaternion = r.as_quat()  
                                quaternion = np.concatenate(([quaternion[3]],quaternion[0:3]))
                                
                                if self.image_pos is not None:
                                    self.vel_img.append((trans - self.image_pos)/(self.quad_env.t_step*(task.frame-self.task_frame_ant)))
                                    self.vel_sens.append(self.quad_sens.velocity_t0)
                                    iv_var = np.mean(np.var(self.vel_img, axis = 0))
                                    vs_var = np.mean(np.var(self.vel_sens, axis = 0))
                                    if iv_var < 0.1 and len(self.vel_img)> 50:
                                        self.quad_sens.velocity_t0 = self.quad_sens.velocity_t0*0.9+self.vel_img[-1]*0.1
                                self.image_pos = trans
                                self.image_quat = quaternion                                
                                self.quad_sens.position_t0 = self.quad_sens.position_t0*0.8+self.image_pos*0.2
                                self.quad_sens.quaternion_t0 = self.quad_sens.quaternion_t0*0.8+self.image_quat*0.2
                                
                                self.draw(img, corners, imgpts)
                                self.task_frame_ant = task.frame
                    # cv.imshow('Drone Camera',np.flipud(cv.cvtColor(img, cv.COLOR_RGB2BGR)))
            self.time_total_img.append(time.time()-time_iter)
        return task.cont