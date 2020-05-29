import cv2 as cv
import numpy as np
import panda3d
import time
from collections import deque
from scipy.spatial.transform import Rotation as R
from tabulate import tabulate

class computer_vision():
    def __init__(self, render, quad_model, quad_env, quad_sens, quad_pos, mydir, IMG_POS_DETER):
        self.IMG_POS_DETER = IMG_POS_DETER
        self.mydir = mydir
        self.quad_model = quad_model
        self.quad_env = quad_env
        self.quad_sens = quad_sens
        self.quad_pos = quad_pos
        self.image_pos = None
        self.vel_sens = deque(maxlen=100)
        self.vel_img = deque(maxlen=100)
        self.calibrated = False
        self.render = render  
        # Load the checkerboard actor
        self.render.checker = self.render.loader.loadModel(self.mydir + '/models/checkerboard.egg')
        self.render.checker.reparentTo(self.render.render)
        self.checker_scale = 0.5
        self.checker_sqr_size = 0.2046
        self.render.checker.setScale(self.checker_scale, self.checker_scale, 1)
        self.render.checker.setPos(3*self.checker_scale*self.checker_sqr_size+0.06, 2.5*self.checker_scale*self.checker_sqr_size+0.06, 0.001)
        self.render.taskMgr.add(self.calibrate, 'Camera Calibration')
        self.criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        self.render.taskMgr.add(self.pos_deter, 'Position Determination')   

        window_size = (self.render.win.getXSize(), self.render.win.getYSize())     
        self.render.buffer = self.render.win.makeTextureBuffer('Buffer', *window_size, None, True)
        self.render.cam_1 = self.render.makeCamera(self.render.buffer)
        self.render.cam_1.setName('cam_1')     
        self.render.cam_1.node().getLens().setFilmSize(36, 24)
        self.render.cam_1.node().getLens().setFocalLength(45)
        self.render.cam_1.reparentTo(self.quad_model)
        self.render.cam_1.setPos(0, 0, 0.01)
        self.render.cam_1.setHpr(0, 270, 0)
        self.time_total_img = []
        self.image_pos = None
        self.vel_sens = deque(maxlen=100)
        self.vel_img = deque(maxlen=100)

            
    def calibrate(self, task):
            if task.frame == 0:
                self.fast = cv.FastFeatureDetector_create()
                self.fast.setThreshold(20)
                self.criteria = (cv.TERM_CRITERIA_EPS + cv.TermCriteria_COUNT, 1, 0.0001) 
                self.nCornersCols = 9
                self.nCornersRows = 6
                self.objp = np.zeros((self.nCornersCols*self.nCornersRows, 3), np.float32)
                self.objp[:,:2] = (np.mgrid[0:self.nCornersCols, 0:self.nCornersRows].T.reshape(-1,2))*self.checker_scale*self.checker_sqr_size
                try: 
                    npzfile = np.load('./config/camera_calibration.npz')
                    self.mtx = npzfile[npzfile.files[0]]
                    self.dist = npzfile[npzfile.files[1]]
                    self.calibrated = True
                    print('Calibration File Loaded')
                    return task.done
                except:
                    print('Could Not Load Calibration File, Calibrating... ')
                    self.calibrated = False
                    self.quad_model.setPos(10,10,10)
                    self.render.cam_pos = []
                    self.objpoints = []
                    self.imgpoints = []
                    
            rand_pos = (np.random.random(3)-0.5)*5
            rand_pos[2] = np.random.random()*3+2
            cam_pos = tuple(rand_pos)
            self.render.cam.reparentTo(self.render.render)
            self.render.cam_1.reparentTo(self.render.render)
            self.render.cam.setPos(*cam_pos)
            self.render.cam.lookAt(self.render.checker)
            self.render.cam_1.setPos(*cam_pos)
            self.render.cam_1.lookAt(self.render.checker)
            ret, image = self.get_image()
            if ret:
                img = cv.cvtColor(image, cv.COLOR_RGBA2BGR)
                self.gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)                
                ret, corners = cv.findChessboardCorners(self.gray, (self.nCornersCols, self.nCornersRows), 
                                                        cv.CALIB_CB_ADAPTIVE_THRESH + cv.CALIB_CB_NORMALIZE_IMAGE + cv.CALIB_CB_FILTER_QUADS+ cv.CALIB_CB_FAST_CHECK)
                if ret:
                    # corners = cv.cornerSubPix(self.gray,corners,(11,11),(-1,-1),self.criteria)
                    self.objpoints.append(self.objp)             
                    self.imgpoints.append(corners)
                    img = cv.drawChessboardCorners(img, (self.nCornersCols, self.nCornersRows), corners, ret)
                    cv.imshow('img',img)
    
            if len(self.objpoints) > 50:
                ret, mtx, dist, rvecs, tvecs = cv.calibrateCamera(self.objpoints, self.imgpoints, self.gray.shape[::-1], None, None)
                if ret:
                    h,  w = img.shape[:2]
                    newcameramtx, roi=cv.getOptimalNewCameraMatrix(mtx,dist,(w,h),1,(w,h))
                    dist = np.array([[0.0, 0.0, 0.0, 0.0, 0.0]]) #Camera Perfeita (Simulada), logo não há distorção
                    dst = cv.undistort(img, mtx, dist, None, newcameramtx)    
                    cv.imshow('img', dst)
                    self.mtx = mtx
                    self.dist = dist
                    print('Calibration Complete')
                    self.calibrated = True
                    np.savez('./config/camera_calibration', mtx, dist)
                    print('Calibration File Saved')
                    self.render.cam.reparentTo(self.render.render)
                    self.render.cam.setPos(self.render.cam_neutral_pos)
                    self.render.cam_1.reparentTo(self.quad_model)
                    self.render.cam_1.setPos(0,0,0.01)
                    return task.done                
                else:
                    return task.cont
            else:
                return task.cont  
        
    def get_image(self):
        tex = self.render.buffer.getTexture()  
        img = tex.getRamImage()
        image = np.frombuffer(img, np.uint8)
        
        if len(image) > 0:
            image = np.reshape(image, (tex.getYSize(), tex.getXSize(), 4))
            image = cv.resize(image, (0,0), fx=0.5, fy=0.5)
            return True, image
        else:
            return False, None
   
    def tabulate_gen(self, real, image, accel, gps, gyro, triad):
        data = []
        header = ['---', 'State', 'Image State', 'Accelerometer State', 'GPS State', 'Gyro State', 'Triad State']
        data_name = ['x', 'y', 'z', 'q0', 'q1', 'q2', 'q3']

        for i in range(3):
            data.append((data_name[i], str(real[i]), str(image[i]), str(accel[i]), str(gps[i]), "0", "0"))
        
        for i in range(4):
            data.append((data_name[i+3], str(real[i+3]), str(image[i+3]), "0", "0", str(gyro[i]), str(triad[i])))
            
        return data, header
    
    def draw(self, img, corners, imgpts):
        corner = tuple(corners[0].ravel())
        img = cv.line(img, corner, tuple(imgpts[0].ravel()), (255, 0, 0), 1)
        img = cv.line(img, corner, tuple(imgpts[1].ravel()), (0, 255, 0), 1)
        img = cv.line(img, corner, tuple(imgpts[2].ravel()), (0, 0, 255), 1)
        return img    
    
    def pos_deter(self, task):
        if self.IMG_POS_DETER:
            time_iter = time.time()
            if self.calibrated and task.frame % 10 == 0:           
                ret, image = self.get_image()
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
                                
                                
                                image_state = np.concatenate((trans.flatten(), quaternion.flatten()))                    
                                data, header = self.tabulate_gen(real_state, image_state, self.quad_pos.pos_accel, self.quad_pos.pos_gps, self.quad_pos.quaternion_gyro, self.quad_pos.quaternion_triad)
                                print(tabulate(data, headers = header, numalign='center', stralign='center', floatfmt='.3f'))
                                print('\n')
                                self.draw(img, corners, imgpts)
                                self.task_frame_ant = task.frame
                    cv.imshow('Drone Camera',np.flipud(cv.cvtColor(img, cv.COLOR_RGB2BGR)))
            self.time_total_img.append(time.time()-time_iter)
        return task.cont