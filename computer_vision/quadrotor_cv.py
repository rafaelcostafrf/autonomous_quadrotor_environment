import cv2 as cv
from collections import deque
import time
import numpy as np
from scipy.spatial.transform import Rotation as R
from computer_vision.detector_setup import detection_setup


class computer_vision():
    def __init__(self, render, quad_model, quad_env, quad_sens, quad_pos, cv_cam, camera_cal, mydir, IMG_POS_DETER):
        
        self.mtx = camera_cal.mtx
        self.dist = camera_cal.dist
        
        self.IMG_POS_DETER = IMG_POS_DETER

        self.quad_env = quad_env
        self.quad_sens = quad_sens
        self.image_pos = None
        self.vel_sens = deque(maxlen=100)
        self.vel_img = deque(maxlen=100)
        self.render = render  
        
        self.fast, self.criteria, self.nCornersCols, self.nCornersRows, self.objp, self.checker_scale, self.checker_sqr_size = detection_setup(render)  
        
        self.render.quad_model.setPos(0, 0, 0)
        self.render.quad_model.setHpr(0, 0, 0)
        
        self.cv_cam = cv_cam
        self.cv_cam.cam.setPos(0, 0, 0.01)
        self.cv_cam.cam.lookAt(0, 0, 0)
        self.cv_cam.cam.reparentTo(self.render.quad_model)

        self.render.taskMgr.add(self.pos_deter, 'position determination algorithm')
    
    def img_show(self, task):
        if task.frame % self.cv_cam.frame_int == 1:           
            ret, image = self.cv_cam.get_image()
            if ret:
                cv.imshow('Drone Camera',image)
                cv.waitKey(1)
        return task.cont
    
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
            if task.frame % 10 == 1:           
                ret, image = self.cv_cam.get_image()
                if ret:
                    img = cv.cvtColor(image, cv.COLOR_RGBA2BGR)
                    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
                    fast_gray = cv.resize(gray, None, fx=1, fy=1)
                    corner_good = self.fast.detect(fast_gray)
                    if len(corner_good) > 50:
                        ret, corners = cv.findChessboardCorners(img, (self.nCornersCols, self.nCornersRows),
                                                                cv.CALIB_CB_ADAPTIVE_THRESH + cv.CALIB_CB_NORMALIZE_IMAGE + cv.CALIB_CB_FILTER_QUADS + cv.CALIB_CB_FAST_CHECK)
                        if ret:
                            ret, rvecs, tvecs = cv.solvePnP(self.objp, corners, self.mtx, self.dist)
                            if ret:
    
                                axis = np.float32([[3,0,0], [0,3,0], [0,0,-3]]).reshape(-1,3)
                                imgpts, jac = cv.projectPoints(axis, rvecs, tvecs, self.mtx, self.dist)
                                imgpts = imgpts.astype(np.int)
        

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