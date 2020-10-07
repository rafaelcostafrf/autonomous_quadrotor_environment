import sys, os
import numpy as np
import argparse

#Panda 3D Imports
from panda3d.core import Filename
from direct.showbase.ShowBase import ShowBase
from panda3d.core import loadPrcFile
loadPrcFile('./config/conf.prc')

#Custom Functions
from environment.position import quad_position
from visual_landing.ppo_training import computer_vision
from visual_landing.ppo_world_setup import world_setup, quad_setup
from models.camera_control import camera_control
from computer_vision.cameras_setup import cameras
from config.menu import menu
from environment.controller.target_parser import episode_n
from direct.gui.OnscreenText import OnscreenText


#!/usr/bin/env python

from PyQt5 import QtCore, QtWidgets
import pyqtgraph as pg
import numpy as np
"""
INF209B − TÓPICOS ESPECIAIS EM PROCESSAMENTO DE SINAIS:

VISAO COMPUTACIONAL

PROJETO

RA: 21201920754
NOME: RAFAEL COSTA FERNANDES
E−MAIL: COSTA.FERNANDES@UFABC.EDU.BR

DESCRIÇÃO:
    Ambiente 3D de simulação do quadrirrotor. 
    Dividido em tarefas, Camera Calibration e Drone position
    Se não haver arquivo de calibração de camera no caminho ./config/camera_calibration.npz, realiza a calibração da câmera tirando 70 fotos aleatórias do padrão xadrez
    
    A bandeira IMG_POS_DETER adiciona a tarefa de determinação de posição por visão computacional, utilizando um marco artificial que já está no ambiente 3D, na origem. 
    A bandeira REAL_STATE_CONTROL determina se o algoritmo de controle será alimentado pelos valores REAIS do estado ou os valores estimados pelos sensores a bordo
    
    O algoritmo não precisa rodar em tempo real, a simulação está totalmente disvinculada da velocidade de renderização do ambiente 3D.
    Se a simulação 3D estiver muito lenta, recomendo mudar a resolução nativa no arquivo de configuração ./config/conf.prc
    Mudando a resolução PROVAVELMENTE será necessário recalibrar a câmera, mas sem problemas adicionais. 
    
    Recomendado rodar em um computador com placa de vídeo. 
    Em um i5-4460 com uma AMD R9-290:
        Taxa de FPS foi cerca de 95 FPS com a bandeira IMG_POS_DETER = False e REAL_STATE_CONTROL = True
        Taxa de FPS foi cerca de 35 FPS com a bandeira IMG_POS_DETER = True e REAL_STATE_CONTROL = False
    
    O algoritmo de detecção de pontos FAST ajudou na velocidade da detecção, mas a performance precisa ser melhorada 
    (principalmente para frames em que o tabuleiro de xadrez aparece parcialmente)
    
    
"""

# REAL STATE CONTROL ELSE BY ESTIMATION METHODS
REAL_CTRL = True

# HOVER FLIGHT ELSE RANDOM INITIAL STATE
HOVER = True

#IMAGE POSITION ESTIMATION
IMG_POS_DETER = True

# NUMBER OF EPISODE TIMESTEPS 
EPISODE_STEPS = 5000

# TOTAL NUMBER OF EPISODES
ERROR_AQS_EPISODES = episode_n()

# PATH TO ERROR DATALOG
ERROR_PATH = './data/hover/' if HOVER else './data/random_initial_state/'

mydir = os.path.abspath(sys.path[0])

mydir = Filename.fromOsSpecific(mydir).getFullpath()

frame_interval = 10
cam_names = ('cam_1', )

class MyApp(ShowBase):
    def __init__(self):
        
        ShowBase.__init__(self)       
        render = self.render
        
        # MODELS SETUP
        world_setup(self, render, mydir)
        quad_setup(self, render, mydir)
        
        # OPENCV CAMERAS SETUP
        self.buffer_cameras = cameras(self, frame_interval, cam_names)  
    
    
        #ONSCREEN TEXT
        self.error_display = OnscreenText(text='', pos=(0, 0.5), scale=0.07, bg=(1, 1, 1, 1))
        self.control_display = OnscreenText(text='', pos=(-0.5, 1), scale=0.07, bg=(1, 1, 1, 1))
    def run_setup(self):
        
        # COMPUTER VISION
        self.cv = computer_vision(self, self.quad_model, self.prop_models, self.buffer_cameras.opencv_cameras[0], self.buffer_cameras.opencv_cam_cal[0], mydir, IMG_POS_DETER, self.error_display, self.control_display)     
        
        # IN ENGINE MENU (REAL TIME CONTROLLER CHANGE)
        # menu(self, self.drone, self.drone.sensor, self.cv, self.buffer_cameras.opencv_cameras[0])
        
        # CAMERA CONTROL
        camera_control(self, self.render) 
                
app = MyApp()
app.run()
