import sys, os

#Panda 3D Imports
from panda3d.core import Filename
from direct.showbase.ShowBase import ShowBase
from panda3d.core import loadPrcFile
loadPrcFile('./config/conf_eval.prc')

#Custom Functions
from visual_landing.rl_eval_worker import quad_worker
from visual_landing.ppo_world_setup import world_setup, quad_setup
from models.camera_control import camera_control
from computer_vision.cameras_setup import cameras
from environment.controller.target_parser import episode_n
from panda3d.core import WindowProperties 

import argparse

parser = argparse.ArgumentParser(description='Child or Mother Process')
parser.add_argument('-c', '--child', action='store_true',
                    help='an integer for the accumulator')
child = parser.parse_args().child


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
camera_size = 84
mydir = os.path.abspath(sys.path[0])

mydir = Filename.fromOsSpecific(mydir).getFullpath()

frame_interval = 10
cam_names = ('cam_1', )

CHILD_PROCESS = False
child = 0

if child:
    print('--------------------')
    print('--------------------')
    print('---CHILD PROCESS----')
    print('--------------------')
    print('--------------------')

else:
    print('--------------------')
    print('--------------------')
    print('---MOTHER PROCESS---')
    print('--------------------')
    print('--------------------')


    
class MyApp(ShowBase):
    def __init__(self):
        
        ShowBase.__init__(self)       
        render = self.render
        
        # MODELS SETUP
        world_setup(self, render, mydir)
        quad_setup(self, render, mydir)
        
        # OPENCV CAMERAS SETUP
        self.buffer_cameras = cameras(self, camera_size, frame_interval, cam_names)  
        
        # self.taskMgr.doMethodLater(60, trace_memory, 'memory check')
        # COMPUTER VISION
        self.ldg_algorithm = quad_worker(self, self.buffer_cameras.opencv_cameras[0], False, child=0)        

        # CAMERA CONTROL
        camera_control(self, self.render) 
        
        #WINDOW NAME
        wn= 'Child process n.'+str(CHILD_PROCESS) if child else 'Mother Process'
        props = WindowProperties( )
        props.setTitle( wn )
        self.win.requestProperties( props )

    def run_setup(self):        
        a = 1        
                
app = MyApp()

app.run()
