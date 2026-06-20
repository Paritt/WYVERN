from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import QMainWindow, QApplication, QLabel, QFileDialog, QAction
from PyQt5.QtGui import QPixmap, QDoubleValidator
import torch
from torch import nn
import timm
import cv2
import numpy as np
import segmentation_models_pytorch as smp
from skimage import io, color
import matplotlib.pyplot as plt
from pydicom import dcmread


class Ui_MainWindow(object):
        def setupUi(self, MainWindow):
                MainWindow.setObjectName("MainWindow")
                import os

                # --- dynamic scaling based on actual screen size ---
                screen_geo = QtWidgets.QApplication.primaryScreen().availableGeometry()
                SW, SH = screen_geo.width(), screen_geo.height()
                BASE_W, BASE_H = 1440, 872
                sw, sh = SW / BASE_W, SH / BASE_H
                sf = min(sw, sh)  # uniform font scale

                def r(x, y, w, h):
                        return QtCore.QRect(int(x * sw), int(y * sh), int(w * sw), int(h * sh))

                def fs(size):
                        return max(8, int(size * sf))

                def font_arial(size, bold=False):
                        f = QtGui.QFont("Arial")
                        f.setPointSize(fs(size))
                        if bold:
                                f.setBold(True)
                                f.setWeight(75)
                        return f
                # ---------------------------------------------------

                icon = QtGui.QIcon()
                icon_path = os.path.join(os.path.dirname(__file__), 'src', 'logo', 'icon.png')
                icon.addPixmap(QtGui.QPixmap(icon_path), QtGui.QIcon.Normal, QtGui.QIcon.Off)
                MainWindow.setWindowIcon(icon)
                MainWindow.setAutoFillBackground(False)
                MainWindow.setStyleSheet("background-color: rgb(0, 0, 0);")
                self.centralwidget = QtWidgets.QWidget(MainWindow)
                self.centralwidget.setObjectName("centralwidget")

                #Select button
                self.select_button = QtWidgets.QPushButton(self.centralwidget)
                self.select_button.setGeometry(r(1030, 320, 331, 56))
                self.select_button.setFont(font_arial(20))
                self.select_button.setStyleSheet("color: rgb(200, 200, 200);\n"
        "background-color: rgba(245, 245, 245, 51);")
                self.select_button.setObjectName("select_button")

                #Analyze button
                self.analyze_button = QtWidgets.QPushButton(self.centralwidget)
                self.analyze_button.setGeometry(r(1030, 390, 331, 56))
                self.analyze_button.setFont(font_arial(20))
                self.analyze_button.setStyleSheet("background-color: rgba(245, 245, 245, 51);\n"
        "color: rgb(200, 200, 200);")
                self.analyze_button.setObjectName("analyze_button")

                #Clear button
                self.clear_button = QtWidgets.QPushButton(self.centralwidget)
                self.clear_button.setGeometry(r(1030, 460, 331, 56))
                self.clear_button.setFont(font_arial(20))
                self.clear_button.setStyleSheet("background-color: rgba(245, 245, 245, 51);\n"
        "color: rgb(200, 200, 200);")
                self.clear_button.setObjectName("clear_button")

                #Img — left panel, full height; content centered via AlignCenter + aspect-ratio scaling
                self.img = QtWidgets.QLabel(self.centralwidget)
                self.img.setGeometry(QtCore.QRect(0, 0, int(941 * sw), SH))
                self.img.setFont(font_arial(24))
                self.img.setStyleSheet("color: rgb(200, 200, 200);")
                self.img.setAlignment(QtCore.Qt.AlignCenter)
                self.img.setObjectName("img")

                #Logo — scaled proportionally with screen, aspect ratio preserved
                self.logo = QtWidgets.QLabel(self.centralwidget)
                logo_path = os.path.join(os.path.dirname(__file__), 'src', 'logo', 'Logo.png')
                logo_pixmap = QtGui.QPixmap(logo_path)
                logo_target = int(241 * sf)
                logo_pixmap = logo_pixmap.scaled(logo_target, logo_target,
                                                  QtCore.Qt.KeepAspectRatio,
                                                  QtCore.Qt.SmoothTransformation)
                logo_w, logo_h = logo_pixmap.width(), logo_pixmap.height()
                right_panel_x = int(941 * sw)
                logo_x = right_panel_x + (SW - right_panel_x - logo_w) // 2
                self.logo.setGeometry(QtCore.QRect(logo_x, int(40 * sh), logo_w, logo_h))
                self.logo.setText("")
                self.logo.setPixmap(logo_pixmap)
                self.logo.setScaledContents(False)
                self.logo.setObjectName("logo")

                # ── Control rows ────────────────────────────────────────────────
                # Layout per group (base coords, 1440×872):
                #   Row A  y=ROW_A   [Checkbox  x=960 w=210 h=38]
                #   Row B  y=ROW_B   [Cutoff label x=960 w=135 h=42]
                #                    [Input        x=1100 w=90 h=42]
                #                    [Result       x=1195 w=225 h=42]  ← same row
                # ────────────────────────────────────────────────────────────────
                ROW1A, ROW1B = 545, 598
                ROW2A, ROW2B = 658, 711
                ROW3A, ROW3B = 771, 824

                CB_X, CB_W, CB_H     = 960, 210, 38
                LBL_X, LBL_W        = 960, 135
                INP_X, INP_W, ROW_H = 1100, 90, 42
                RES_X, RES_W, RES_H = 1195, 225, 52

                #Inspire checkbox
                self.Inspiration_checkBox = QtWidgets.QCheckBox(self.centralwidget)
                self.Inspiration_checkBox.setGeometry(r(CB_X, ROW1A, CB_W, CB_H))
                self.Inspiration_checkBox.setFont(font_arial(17, bold=True))
                self.Inspiration_checkBox.setChecked(True)
                self.Inspiration_checkBox.setStyleSheet("color: rgb(200, 200, 200)")
                self.Inspiration_checkBox.setObjectName("Inspiration_checkBox")

                self.ROL_label = QtWidgets.QLabel(self.centralwidget)
                self.ROL_label.setGeometry(r(LBL_X, ROW1B, LBL_W, ROW_H))
                self.ROL_label.setFont(font_arial(16))
                self.ROL_label.setStyleSheet("color: rgb(200, 200, 200)")
                self.ROL_label.setObjectName("ROL_label")

                self.cutoff_rol = QtWidgets.QLineEdit(self.centralwidget)
                self.cutoff_rol.setGeometry(r(INP_X, ROW1B, INP_W, ROW_H))
                self.cutoff_rol.setFont(font_arial(16))
                self.cutoff_rol.setStyleSheet("color: rgb(200, 200, 200)")
                self.cutoff_rol.setAlignment(QtCore.Qt.AlignCenter)
                self.cutoff_rol.setObjectName("cutoff_rol")

                self.ROL = QtWidgets.QLabel(self.centralwidget)
                self.ROL.setGeometry(r(RES_X, ROW1B - 5, RES_W, RES_H))
                self.ROL.setFont(font_arial(14))
                self.ROL.setFocusPolicy(QtCore.Qt.NoFocus)
                self.ROL.setStyleSheet("color: rgb(0, 0, 0);")
                self.ROL.setText("")
                self.ROL.setScaledContents(False)
                self.ROL.setAlignment(QtCore.Qt.AlignVCenter | QtCore.Qt.AlignLeft)
                self.ROL.setObjectName("ROL")

                #Lt checkbox
                self.Lt_checkBox = QtWidgets.QCheckBox(self.centralwidget)
                self.Lt_checkBox.setGeometry(r(CB_X, ROW2A, CB_W, CB_H))
                self.Lt_checkBox.setFont(font_arial(17, bold=True))
                self.Lt_checkBox.setChecked(True)
                self.Lt_checkBox.setStyleSheet("color: rgb(200, 200, 200)")
                self.Lt_checkBox.setObjectName("Lt_checkBox")

                self.Lt_Label = QtWidgets.QLabel(self.centralwidget)
                self.Lt_Label.setGeometry(r(LBL_X, ROW2B, LBL_W, ROW_H))
                self.Lt_Label.setFont(font_arial(16))
                self.Lt_Label.setStyleSheet("color: rgb(200, 200, 200)")
                self.Lt_Label.setObjectName("Lt_Label")

                self.cutoff_LtAlpha = QtWidgets.QLineEdit(self.centralwidget)
                self.cutoff_LtAlpha.setGeometry(r(INP_X, ROW2B, INP_W, ROW_H))
                self.cutoff_LtAlpha.setFont(font_arial(16))
                self.cutoff_LtAlpha.setStyleSheet("color: rgb(200, 200, 200)")
                self.cutoff_LtAlpha.setAlignment(QtCore.Qt.AlignCenter)
                self.cutoff_LtAlpha.setObjectName("cutoff_LtAlpha")

                self.Lt_Alpha = QtWidgets.QLabel(self.centralwidget)
                self.Lt_Alpha.setGeometry(r(RES_X, ROW2B - 5, RES_W, RES_H))
                self.Lt_Alpha.setFont(font_arial(14))
                self.Lt_Alpha.setFocusPolicy(QtCore.Qt.NoFocus)
                self.Lt_Alpha.setStyleSheet("color: rgb(0, 0, 0);")
                self.Lt_Alpha.setText("")
                self.Lt_Alpha.setScaledContents(False)
                self.Lt_Alpha.setAlignment(QtCore.Qt.AlignVCenter | QtCore.Qt.AlignLeft)
                self.Lt_Alpha.setObjectName("Lt_Alpha")

                #Rt checkbox
                self.Rt_checkBox = QtWidgets.QCheckBox(self.centralwidget)
                self.Rt_checkBox.setGeometry(r(CB_X, ROW3A, CB_W, CB_H))
                self.Rt_checkBox.setFont(font_arial(17, bold=True))
                self.Rt_checkBox.setChecked(True)
                self.Rt_checkBox.setStyleSheet("color: rgb(200, 200, 200)")
                self.Rt_checkBox.setObjectName("Rt_checkBox")

                self.Rt_label = QtWidgets.QLabel(self.centralwidget)
                self.Rt_label.setGeometry(r(LBL_X, ROW3B, LBL_W, ROW_H))
                self.Rt_label.setFont(font_arial(16))
                self.Rt_label.setStyleSheet("color: rgb(200, 200, 200)")
                self.Rt_label.setObjectName("Rt_label")

                self.cutoff_RtAlpha = QtWidgets.QLineEdit(self.centralwidget)
                self.cutoff_RtAlpha.setGeometry(r(INP_X, ROW3B, INP_W, ROW_H))
                self.cutoff_RtAlpha.setFont(font_arial(16))
                self.cutoff_RtAlpha.setStyleSheet("color: rgb(200, 200, 200)")
                self.cutoff_RtAlpha.setAlignment(QtCore.Qt.AlignCenter)
                self.cutoff_RtAlpha.setObjectName("cutoff_RtAlpha")

                self.Rt_Alpha = QtWidgets.QLabel(self.centralwidget)
                self.Rt_Alpha.setGeometry(r(RES_X, ROW3B - 5, RES_W, RES_H))
                self.Rt_Alpha.setFont(font_arial(14))
                self.Rt_Alpha.setFocusPolicy(QtCore.Qt.NoFocus)
                self.Rt_Alpha.setStyleSheet("color: rgb(0, 0, 0);")
                self.Rt_Alpha.setText("")
                self.Rt_Alpha.setScaledContents(False)
                self.Rt_Alpha.setAlignment(QtCore.Qt.AlignVCenter | QtCore.Qt.AlignLeft)
                self.Rt_Alpha.setObjectName("Rt_Alpha")

                MainWindow.setCentralWidget(self.centralwidget)
                self.menubar = QtWidgets.QMenuBar(MainWindow)
                self.menubar.setGeometry(r(0, 0, 1440, 24))
                self.menubar.setObjectName("menubar")
                MainWindow.setMenuBar(self.menubar)
                self.statusbar = QtWidgets.QStatusBar(MainWindow)
                self.statusbar.setObjectName("statusbar")
                MainWindow.setStatusBar(self.statusbar)

                self.retranslateUi(MainWindow)
                QtCore.QMetaObject.connectSlotsByName(MainWindow)
                
                ##########################################################################################################################
                ####################################################### Connect ##########################################################
                ##########################################################################################################################
                
                self.select_button.clicked.connect(self.select_img)
                self.clear_button.clicked.connect(self.clear)
                self.analyze_button.clicked.connect(self.analyze)
                self.Lt_checkBox.stateChanged.connect(self.on_checkbox_state_changed)
                self.Rt_checkBox.stateChanged.connect(self.on_checkbox_state_changed)
                self.Inspiration_checkBox.stateChanged.connect(self.on_checkbox_state_changed)
                
                ##########################################################################################################################
                ##########################################################################################################################

        def retranslateUi(self, MainWindow):
                _translate = QtCore.QCoreApplication.translate
                MainWindow.setWindowTitle(_translate("MainWindow", "WYVERN v2026"))
                self.select_button.setText(_translate("MainWindow", "Select Image"))
                self.analyze_button.setText(_translate("MainWindow", "Analyze"))
                self.clear_button.setText(_translate("MainWindow", "Clear"))
                self.img.setText(_translate("MainWindow", "Please choose image"))
                self.cutoff_rol.setText(_translate("MainWindow", "83.92"))
                self.ROL_label.setText(_translate("MainWindow", "Cutoff ROL:"))
                self.Lt_Label.setText(_translate("MainWindow", "Cutoff Alpha:"))
                self.cutoff_LtAlpha.setText(_translate("MainWindow", "-0.2"))
                self.Inspiration_checkBox.setText(_translate("MainWindow", "Inspiration"))
                self.Lt_checkBox.setText(_translate("MainWindow", "Lt. Rotation"))
                self.Rt_checkBox.setText(_translate("MainWindow", "Rt. Rotation"))
                self.Rt_label.setText(_translate("MainWindow", "Cutoff Alpha:"))
                self.cutoff_RtAlpha.setText(_translate("MainWindow", "0.2"))
                
                ##########################################################################################################################
                ################################################## Load Rib & Lung model #################################################
                ##########################################################################################################################
                lung_model_path = 'src/models/lung_MAnet_20260209_042722_best.pth'
                rib_model_path = 'src/models/rib_MAnet_20260209_034825_best.pth'
                self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

                class MAnetWithDropout(nn.Module):
                        """MAnet wrapper that applies spatial dropout on decoder features."""

                        def __init__(self, dropout_p=0.2, **manet_kwargs):
                                super().__init__()
                                self.model = smp.MAnet(**manet_kwargs)
                                self.dropout = nn.Dropout2d(p=dropout_p)
                                self.dropout_p = dropout_p
                                
                        def name(self):
                                return f"MAnetWithDropout_{self.dropout_p}"

                        def forward(self, x):
                                features = self.model.encoder(x)
                                decoder_output = self.model.decoder(*features)
                                decoder_output = self.dropout(decoder_output)
                                masks = self.model.segmentation_head(decoder_output)

                                if self.model.classification_head is not None:
                                        labels = self.model.classification_head(features[-1])
                                        return masks, labels

                                return masks
                SIZE_X = 512 
                SIZE_Y = 512
                ENCODER = 'resnet34'
                ENCODER_WEIGHTS = 'imagenet'
                activation = None

                self.rib_model = MAnetWithDropout(
                                        dropout_p=0.5,
                                        encoder_name=ENCODER,
                                        encoder_weights=ENCODER_WEIGHTS,
                                        in_channels=3,
                                        classes=3
                                        )
                self.rib_model.to(self.device)
                self.rib_pre_dict = torch.load(rib_model_path, map_location=self.device)
                self.rib_model.load_state_dict(self.rib_pre_dict['model'])
                self.rib_model.eval()
                
                self.lung_model = MAnetWithDropout(
                                        dropout_p=0.5,
                                        encoder_name=ENCODER,
                                        encoder_weights=ENCODER_WEIGHTS,
                                        in_channels=3,
                                        classes=3
                                        )
                self.lung_model.to(self.device)
                self.lung_pre_dict = torch.load(lung_model_path, map_location=self.device)
                self.lung_model.load_state_dict(self.lung_pre_dict['model'])
                self.lung_model.eval()
                
                ##########################################################################################################################
                ################################################### Load Rotation model ##################################################
                ##########################################################################################################################
                class HRNetCustomModel(nn.Module):
                        def __init__(self):
                                super(HRNetCustomModel, self).__init__()
                                
                                # HRNet Backbone (ใช้จาก timm)
                                self.hrnet = timm.create_model('hrnet_w48', pretrained=True, features_only=True)
                                
                                # Additional Fully Connected Layers
                                self.flatten = nn.Flatten()
                                self.fc1 = nn.Linear(1024 * 16 * 16, 1024)  # ปรับขนาดให้ตรงกับ output feature map
                                self.fc2 = nn.Linear(1024, 512)
                                self.fc3 = nn.Linear(512, 256)
                                self.fc4 = nn.Linear(256, 128)
                                self.fc5 = nn.Linear(128, 64)
                                self.fc6 = nn.Linear(64, 3)

                        def forward(self, x):
                                features = self.hrnet(x)
                                x = features[-1]  # ใช้ feature map สุดท้าย
                                x = self.flatten(x)
                                x = torch.relu(self.fc1(x))
                                x = torch.relu(self.fc2(x))
                                x = torch.relu(self.fc3(x))
                                x = torch.relu(self.fc4(x))
                                x = torch.relu(self.fc5(x))
                                return self.fc6(x)
                
                self.rotation_model = HRNetCustomModel().to(self.device)
                self.rotation_model.load_state_dict(torch.load('src/models/rotation_model.pth', weights_only=True,map_location=self.device))
                self.rotation_model.to(self.device)
                self.rotation_model.eval()
                ##########################################################################################################################
                ##########################################################################################################################

        ##########################################################################################################################
        ######################################################## FUNCTION ########################################################
        ##########################################################################################################################
        def select_img(self):
                self.imagePath, _ = QFileDialog.getOpenFileName()
                if self.imagePath[-1] == 'm':
                        img = dcmread(self.imagePath)
                        img = img.pixel_array
                        #save
                        plt.clf()
                        plt.imshow(img, cmap='gray')
                        plt.axis("off")
                        plt.savefig('result/dcm.png', bbox_inches='tight', facecolor='black')
                        #Show
                        self._show_pixmap('result/dcm.png')
                else:
                        self._show_pixmap(QPixmap(self.imagePath))

        def clear(self):
                self.img.setText("Please select image")
                self.ROL.setText(" ")
                self.Lt_Alpha.setText(" ")
                self.Rt_Alpha.setText(" ")

        def analyze(self):

                def np_to_torch(np_array):      return torch.from_numpy(np_array).float()
                def torch_to_np(torch_array):   return np.squeeze(torch_array.detach().cpu().numpy())
                def preprocess(img):
                        preprocessing_fn = smp.encoders.get_preprocessing_fn('resnext50_32x4d', 'imagenet')
                        img_r = img.astype(np.float64)
                        img = preprocessing_fn(img)
                        image = img.astype(np.float64)
                        image = np.transpose(image, (2, 0, 1))
                        img_input = np.expand_dims(image, 0)
                        img_input = np_to_torch(img_input).to(self.device)
                        
                        image_r = np.transpose(img_r, (2, 0, 1))
                        img_input_r = np.expand_dims(image_r, 0)
                        img_input_r = np_to_torch(img_input_r).to(self.device)
                        return img_input, img_input_r
                
                def cal_distance(point):
                        x1, x2, x3 = point
                        distance_left = round(x2 - x3)
                        distance_right = round(x3 - x1)
                        return distance_left, distance_right

                def cal_alpha(distance_left, distance_right):
                        alpha = (distance_right - distance_left) / (distance_right + distance_left)
                        return round(alpha, 3)  # ปัดค่า alpha ให้เหลือทศนิยม 3 ตำแหน่ง
                ##########################################################################################################################
                ######################################################## Preprocess ######################################################
                ##########################################################################################################################
                
                if self.imagePath[-1] == 'm':
                        SIZE_X = 512 
                        SIZE_Y = 512
                        ds = dcmread(self.imagePath)
                        pixel_array = ds.pixel_array
                        pixel_array = pixel_array.astype('float64')
                        pixel_array /= np.max(pixel_array)
                        image = cv2.resize(pixel_array, (SIZE_Y, SIZE_X), interpolation = cv2.INTER_NEAREST)
                        image = np.repeat(image[:,:,np.newaxis],3,2)
                        img_input, img_input_r = preprocess(image)
                else:
                        SIZE_X = 512 
                        SIZE_Y = 512
                        image = cv2.imread(self.imagePath, 1) #Read in BGR mode (1)
                        image = cv2.resize(image, (SIZE_Y, SIZE_X), interpolation = cv2.INTER_NEAREST)
                        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                        image = image/255
                        img_input, img_input_r = preprocess(image)
                        
                
                ##########################################################################################################################
                ######################################################## Prediction ######################################################
                ##########################################################################################################################
                
                r_y_pred = self.rib_model(img_input)
                r_y_pred_np = torch_to_np(r_y_pred)
                r_y_pred_argmax = np.argmax(r_y_pred_np, axis=0)
                l_y_pred = self.lung_model(img_input)
                l_y_pred_np = torch_to_np(l_y_pred)
                l_y_pred_argmax = np.argmax(l_y_pred_np, axis=0)
                rotate_pred = self.rotation_model(img_input_r)
                rotate_pred = rotate_pred[0, :].detach().cpu().numpy()
                distance_left, distance_right = cal_distance(rotate_pred)  # ย้าย Tensor ไป CPU ก่อน
                self.alpha = cal_alpha(distance_left, distance_right)  # คำนวณค่า alpha
                L_alpha_cutoff = float(self.cutoff_LtAlpha.text())
                R_alpha_cutoff = float(self.cutoff_RtAlpha.text())
                
                #Overlay
                l_y_pred_argmax[l_y_pred_argmax>0] = 1
                r_y_pred_argmax[r_y_pred_argmax>0] = 2
                r_l_y_pred_argmax = r_y_pred_argmax + l_y_pred_argmax
                overlay = color.label2rgb(r_l_y_pred_argmax,image,colors=[(0,0,100),(100,0,0),(0,100,0)],alpha=0.005, bg_label=0, bg_color=None)
                self.overlay = overlay
                self.image = image
                self.rotate_pred = rotate_pred
                #save
                inspire_check_status = self.Inspiration_checkBox.isChecked()
                lt_check_status = self.Lt_checkBox.isChecked()
                rt_check_status = self.Rt_checkBox.isChecked()
                plt.clf()
                
                if inspire_check_status == True and lt_check_status == True and rt_check_status == True:
                        plt.imshow(overlay)
                        plt.axis('off')
                        y_values = np.linspace(0, image.shape[0], 3)
                        if self.alpha>R_alpha_cutoff:
                                plt.plot([rotate_pred[0]] * len(y_values), y_values, "r-") # Right
                                plt.hlines(y=image.shape[0] // 4 + 10, xmin=rotate_pred[0], xmax=rotate_pred[2], colors="red", linestyles="dashed", label="Distance Right")
                                
                        else:
                                plt.plot([rotate_pred[0]] * len(y_values), y_values, "g-") # Right
                                plt.hlines(y=image.shape[0] // 4 + 10, xmin=rotate_pred[0], xmax=rotate_pred[2], colors="green", linestyles="dashed", label="Distance Right")
                        if self.alpha<L_alpha_cutoff:
                                plt.plot([rotate_pred[1]] * len(y_values), y_values, "r-") # Left
                                plt.hlines(y=image.shape[0] // 4, xmin=rotate_pred[2], xmax=rotate_pred[1], colors="red", linestyles="dashed", label="Distance Left")
                        else:
                                plt.plot([rotate_pred[1]] * len(y_values), y_values, "g-") # Left
                                plt.hlines(y=image.shape[0] // 4, xmin=rotate_pred[2], xmax=rotate_pred[1], colors="green", linestyles="dashed", label="Distance Left")
                        plt.plot([rotate_pred[2]] * len(y_values), y_values, "w-")
                        #plt.legend()
                        plt.savefig('result/result.png', bbox_inches='tight', facecolor='black')
                        #Show
                        self._show_pixmap('result/result.png')
                elif inspire_check_status == False and lt_check_status == True and rt_check_status == True:
                        plt.imshow(image)
                        plt.axis('off')
                        y_values = np.linspace(0, image.shape[0], 3)
                        if self.alpha>R_alpha_cutoff:
                                plt.plot([rotate_pred[0]] * len(y_values), y_values, "r-") # Right
                                plt.hlines(y=image.shape[0] // 4 + 10, xmin=rotate_pred[0], xmax=rotate_pred[2], colors="red", linestyles="dashed", label="Distance Right")
                                
                        else:
                                plt.plot([rotate_pred[0]] * len(y_values), y_values, "g-") # Right
                                plt.hlines(y=image.shape[0] // 4 + 10, xmin=rotate_pred[0], xmax=rotate_pred[2], colors="green", linestyles="dashed", label="Distance Right")
                        if self.alpha<L_alpha_cutoff:
                                plt.plot([rotate_pred[1]] * len(y_values), y_values, "r-") # Left
                                plt.hlines(y=image.shape[0] // 4, xmin=rotate_pred[2], xmax=rotate_pred[1], colors="red", linestyles="dashed", label="Distance Left")
                        else:
                                plt.plot([rotate_pred[1]] * len(y_values), y_values, "g-") # Left
                                plt.hlines(y=image.shape[0] // 4, xmin=rotate_pred[2], xmax=rotate_pred[1], colors="green", linestyles="dashed", label="Distance Left")
                        plt.plot([rotate_pred[2]] * len(y_values), y_values, "w-")
                        #plt.legend()
                        plt.savefig('result/result.png', bbox_inches='tight', facecolor='black')
                        #Show
                        self._show_pixmap('result/result.png')
                        
                elif inspire_check_status == False and lt_check_status == False and rt_check_status == True:
                        plt.imshow(image)
                        plt.axis('off')
                        if self.alpha>R_alpha_cutoff:
                                plt.plot([rotate_pred[0]] * len(y_values), y_values, "r-") # Right
                                plt.hlines(y=image.shape[0] // 4 + 10, xmin=rotate_pred[0], xmax=rotate_pred[2], colors="red", linestyles="dashed", label="Distance Right")
                                
                        else:
                                plt.plot([rotate_pred[0]] * len(y_values), y_values, "g-") # Right
                                plt.hlines(y=image.shape[0] // 4 + 10, xmin=rotate_pred[0], xmax=rotate_pred[2], colors="green", linestyles="dashed", label="Distance Right")
                        plt.plot([rotate_pred[2]] * len(y_values), y_values, "w-")
                        #plt.legend()
                        plt.savefig('result/result.png', bbox_inches='tight', facecolor='black')
                        #Show
                        self._show_pixmap('result/result.png')
                        
                elif inspire_check_status == False and lt_check_status == True and rt_check_status == False:
                        plt.imshow(image)
                        plt.axis('off')
                        y_values = np.linspace(0, image.shape[0], 3)
                        if self.alpha<L_alpha_cutoff:
                                plt.plot([rotate_pred[1]] * len(y_values), y_values, "r-") # Left
                                plt.hlines(y=image.shape[0] // 4, xmin=rotate_pred[2], xmax=rotate_pred[1], colors="red", linestyles="dashed", label="Distance Left")
                        else:
                                plt.plot([rotate_pred[1]] * len(y_values), y_values, "g-") # Left
                                plt.hlines(y=image.shape[0] // 4, xmin=rotate_pred[2], xmax=rotate_pred[1], colors="green", linestyles="dashed", label="Distance Left")
                        plt.plot([rotate_pred[2]] * len(y_values), y_values, "w-")
                        #plt.legend()
                        plt.savefig('result/result.png', bbox_inches='tight', facecolor='black')
                        #Show
                        self._show_pixmap('result/result.png')
                        
                elif inspire_check_status == True and lt_check_status == False and rt_check_status == True:
                        plt.imshow(overlay)
                        plt.axis('off')
                        y_values = np.linspace(0, image.shape[0], 3)
                        if self.alpha>R_alpha_cutoff:
                                plt.plot([rotate_pred[0]] * len(y_values), y_values, "r-") # Right
                                plt.hlines(y=image.shape[0] // 4 + 10, xmin=rotate_pred[0], xmax=rotate_pred[2], colors="red", linestyles="dashed", label="Distance Right")
                                
                        else:
                                plt.plot([rotate_pred[0]] * len(y_values), y_values, "g-") # Right
                                plt.hlines(y=image.shape[0] // 4 + 10, xmin=rotate_pred[0], xmax=rotate_pred[2], colors="green", linestyles="dashed", label="Distance Right")
                        plt.plot([rotate_pred[2]] * len(y_values), y_values, "w-")
                        #plt.legend()
                        plt.savefig('result/result.png', bbox_inches='tight', facecolor='black')
                        #Show
                        self._show_pixmap('result/result.png')
                        
                elif inspire_check_status == True and lt_check_status == True and rt_check_status == False:
                        plt.imshow(overlay)
                        plt.axis('off')
                        y_values = np.linspace(0, image.shape[0], 3)
                        if self.alpha<L_alpha_cutoff:
                                plt.plot([rotate_pred[1]] * len(y_values), y_values, "r-") # Left
                                plt.hlines(y=image.shape[0] // 4, xmin=rotate_pred[2], xmax=rotate_pred[1], colors="red", linestyles="dashed", label="Distance Left")
                        else:
                                plt.plot([rotate_pred[1]] * len(y_values), y_values, "g-") # Left
                                plt.hlines(y=image.shape[0] // 4, xmin=rotate_pred[2], xmax=rotate_pred[1], colors="green", linestyles="dashed", label="Distance Left")
                        plt.plot([rotate_pred[2]] * len(y_values), y_values, "w-")
                        #plt.legend()
                        plt.savefig('result/result.png', bbox_inches='tight', facecolor='black')
                        #Show
                        self._show_pixmap('result/result.png')
                        
                elif inspire_check_status == True and lt_check_status == False and rt_check_status == False:
                        plt.imshow(self.overlay)
                        plt.axis('off')        
                        plt.savefig('result/result.png', bbox_inches='tight', facecolor='black')
                        #Show
                        self._show_pixmap('result/result.png')
                        
                else:
                        plt.imshow(image)
                        plt.axis('off')
                        plt.savefig('result/result.png', bbox_inches='tight', facecolor='black')
                        #Show
                        self._show_pixmap('result/result.png')
                        
                #Calculate IoU
                union = np.count_nonzero(r_l_y_pred_argmax == 2) + np.count_nonzero(r_l_y_pred_argmax == 3)
                intersec = np.count_nonzero(r_l_y_pred_argmax == 3)
                rol = intersec/union * 100
                #Show IoU
                cutoff = float(self.cutoff_rol.text())
                if rol>cutoff:
                        self.ROL.setText(f'Full inspiration\n(ROL: {rol:.2f}%)')
                        self.ROL.setStyleSheet("color: rgb(100, 250, 100)")
                else:
                        self.ROL.setText(f'Not full inspiration\n(ROL: {rol:.2f}%)')
                        self.ROL.setStyleSheet("color: rgb(250, 100, 100)")
                #Show Alpha
                
                if self.alpha>L_alpha_cutoff and self.alpha<R_alpha_cutoff:
                        self.Lt_Alpha.setText(f'No Lt. Rotation\n(Alpha: {self.alpha:.2f})')
                        self.Lt_Alpha.setStyleSheet("color: rgb(100, 250, 100)") # Green
                        
                        self.Rt_Alpha.setText(f'No Rt. Rotation\n(Alpha: {self.alpha:.2f})')
                        self.Rt_Alpha.setStyleSheet("color: rgb(100, 250, 100)") # Green
                        
                elif self.alpha<L_alpha_cutoff and self.alpha<R_alpha_cutoff:
                        self.Lt_Alpha.setText(f'Lt. Rotation\n(Alpha: {self.alpha:.2f})')
                        self.Lt_Alpha.setStyleSheet("color: rgb(250, 100, 100)") # Red
                        
                        self.Rt_Alpha.setText(f'No Rt. Rotation\n(Alpha: {self.alpha:.2f})')
                        self.Rt_Alpha.setStyleSheet("color: rgb(100, 250, 100)") # Green
                        
                elif self.alpha>L_alpha_cutoff and self.alpha>R_alpha_cutoff:
                        self.Lt_Alpha.setText(f'No Lt. Rotation\n(Alpha: {self.alpha:.2f})')
                        self.Lt_Alpha.setStyleSheet("color: rgb(100, 250, 100)")  # Green
                        
                        self.Rt_Alpha.setText(f'Rt. Rotation\n(Alpha: {self.alpha:.2f})')
                        self.Rt_Alpha.setStyleSheet("color: rgb(250, 100, 100)") # Red

                elif self.alpha<L_alpha_cutoff and self.alpha>R_alpha_cutoff:
                        self.Lt_Alpha.setText(f'Lt. Rotation\n(Alpha: {self.alpha:.2f})')
                        self.Lt_Alpha.setStyleSheet("color: rgb(250, 100, 100)") # Red
                        
                        self.Rt_Alpha.setText(f'Rt. Rotation\n(Alpha: {self.alpha:.2f})')
                        self.Rt_Alpha.setStyleSheet("color: rgb(250, 100, 100)") # Red
                        
        def on_checkbox_state_changed(self):
                
                inspire_check_status = self.Inspiration_checkBox.isChecked()
                lt_check_status = self.Lt_checkBox.isChecked()
                rt_check_status = self.Rt_checkBox.isChecked()
                L_alpha_cutoff = float(self.cutoff_LtAlpha.text())
                R_alpha_cutoff = float(self.cutoff_RtAlpha.text())
                plt.clf()
                
                if self.Inspiration_checkBox.isChecked() or self.Lt_checkBox.isChecked() or self.Rt_checkBox.isChecked():
                        try:
                                
                                if inspire_check_status == True and lt_check_status == True and rt_check_status == True:
                                        plt.imshow(self.overlay)
                                        plt.axis('off')
                                        y_values = np.linspace(0, self.image.shape[0], 3)
                                        if self.alpha>R_alpha_cutoff:
                                                plt.plot([self.rotate_pred[0]] * len(y_values), y_values, "r-") # Right
                                                plt.hlines(y=self.image.shape[0] // 4 + 10, xmin=self.rotate_pred[0], xmax=self.rotate_pred[2], colors="red", linestyles="dashed", label="Distance Right")
                                                
                                        else:
                                                plt.plot([self.rotate_pred[0]] * len(y_values), y_values, "g-") # Right
                                                plt.hlines(y=self.image.shape[0] // 4 + 10, xmin=self.rotate_pred[0], xmax=self.rotate_pred[2], colors="green", linestyles="dashed", label="Distance Right")
                                        if self.alpha<L_alpha_cutoff:
                                                plt.plot([self.rotate_pred[1]] * len(y_values), y_values, "r-") # Left
                                                plt.hlines(y=self.image.shape[0] // 4, xmin=self.rotate_pred[2], xmax=self.rotate_pred[1], colors="red", linestyles="dashed", label="Distance Left")
                                        else:
                                                plt.plot([self.rotate_pred[1]] * len(y_values), y_values, "g-") # Left
                                                plt.hlines(y=self.image.shape[0] // 4, xmin=self.rotate_pred[2], xmax=self.rotate_pred[1], colors="green", linestyles="dashed", label="Distance Left")
                                        plt.plot([self.rotate_pred[2]] * len(y_values), y_values, "w-")
                                        #plt.legend()
                                        plt.savefig('result/result.png', bbox_inches='tight', facecolor='black')
                                        #Show
                                        self._show_pixmap('result/result.png')
                                elif inspire_check_status == False and lt_check_status == True and rt_check_status == True:
                                        plt.imshow(self.image)
                                        plt.axis('off')
                                        y_values = np.linspace(0, self.image.shape[0], 3)
                                        if self.alpha>R_alpha_cutoff:
                                                plt.plot([self.rotate_pred[0]] * len(y_values), y_values, "r-") # Right
                                                plt.hlines(y=self.image.shape[0] // 4 + 10, xmin=self.rotate_pred[0], xmax=self.rotate_pred[2], colors="red", linestyles="dashed", label="Distance Right")
                                                
                                        else:
                                                plt.plot([self.rotate_pred[0]] * len(y_values), y_values, "g-") # Right
                                                plt.hlines(y=self.image.shape[0] // 4 + 10, xmin=self.rotate_pred[0], xmax=self.rotate_pred[2], colors="green", linestyles="dashed", label="Distance Right")
                                        if self.alpha<L_alpha_cutoff:
                                                plt.plot([self.rotate_pred[1]] * len(y_values), y_values, "r-") # Left
                                                plt.hlines(y=self.image.shape[0] // 4, xmin=self.rotate_pred[2], xmax=self.rotate_pred[1], colors="red", linestyles="dashed", label="Distance Left")
                                        else:
                                                plt.plot([self.rotate_pred[1]] * len(y_values), y_values, "g-") # Left
                                                plt.hlines(y=self.image.shape[0] // 4, xmin=self.rotate_pred[2], xmax=self.rotate_pred[1], colors="green", linestyles="dashed", label="Distance Left")
                                        plt.plot([self.rotate_pred[2]] * len(y_values), y_values, "w-")
                                        #plt.legend()
                                        plt.savefig('result/result.png', bbox_inches='tight', facecolor='black')
                                        #Show
                                        self._show_pixmap('result/result.png')
                                        
                                elif inspire_check_status == False and lt_check_status == False and rt_check_status == True:
                                        plt.imshow(self.image)
                                        plt.axis('off')
                                        y_values = np.linspace(0, self.image.shape[0], 3)
                                        if self.alpha>R_alpha_cutoff:
                                                plt.plot([self.rotate_pred[0]] * len(y_values), y_values, "r-") # Right
                                                plt.hlines(y=self.image.shape[0] // 4 + 10, xmin=self.rotate_pred[0], xmax=self.rotate_pred[2], colors="red", linestyles="dashed", label="Distance Right")
                                                
                                        else:
                                                plt.plot([self.rotate_pred[0]] * len(y_values), y_values, "g-") # Right
                                                plt.hlines(y=self.image.shape[0] // 4 + 10, xmin=self.rotate_pred[0], xmax=self.rotate_pred[2], colors="green", linestyles="dashed", label="Distance Right")
                                        plt.plot([self.rotate_pred[2]] * len(y_values), y_values, "w-")
                                        #plt.legend()
                                        plt.savefig('result/result.png', bbox_inches='tight', facecolor='black')
                                        #Show
                                        self._show_pixmap('result/result.png')
                                        
                                elif inspire_check_status == False and lt_check_status == True and rt_check_status == False:
                                        plt.imshow(self.image)
                                        plt.axis('off')
                                        y_values = np.linspace(0, self.image.shape[0], 3)
                                        if self.alpha<L_alpha_cutoff:
                                                plt.plot([self.rotate_pred[1]] * len(y_values), y_values, "r-") # Left
                                                plt.hlines(y=self.image.shape[0] // 4, xmin=self.rotate_pred[2], xmax=self.rotate_pred[1], colors="red", linestyles="dashed", label="Distance Left")
                                        else:
                                                plt.plot([self.rotate_pred[1]] * len(y_values), y_values, "g-") # Left
                                                plt.hlines(y=self.image.shape[0] // 4, xmin=self.rotate_pred[2], xmax=self.rotate_pred[1], colors="green", linestyles="dashed", label="Distance Left")
                                        plt.plot([self.rotate_pred[2]] * len(y_values), y_values, "w-")
                                        #plt.legend()
                                        plt.savefig('result/result.png', bbox_inches='tight', facecolor='black')
                                        #Show
                                        self._show_pixmap('result/result.png')
                                        
                                elif inspire_check_status == True and lt_check_status == False and rt_check_status == True:
                                        plt.imshow(self.overlay)
                                        plt.axis('off')
                                        y_values = np.linspace(0, self.image.shape[0], 3)
                                        if self.alpha>R_alpha_cutoff:
                                                plt.plot([self.rotate_pred[0]] * len(y_values), y_values, "r-") # Right
                                                plt.hlines(y=self.image.shape[0] // 4 + 10, xmin=self.rotate_pred[0], xmax=self.rotate_pred[2], colors="red", linestyles="dashed", label="Distance Right")
                                                
                                        else:
                                                plt.plot([self.rotate_pred[0]] * len(y_values), y_values, "g-") # Right
                                                plt.hlines(y=self.image.shape[0] // 4 + 10, xmin=self.rotate_pred[0], xmax=self.rotate_pred[2], colors="green", linestyles="dashed", label="Distance Right")
                                        plt.plot([self.rotate_pred[2]] * len(y_values), y_values, "w-")
                                        #plt.legend()
                                        plt.savefig('result/result.png', bbox_inches='tight', facecolor='black')
                                        #Show
                                        self._show_pixmap('result/result.png')
                                        
                                elif inspire_check_status == True and lt_check_status == True and rt_check_status == False:
                                        plt.imshow(self.overlay)
                                        plt.axis('off')
                                        y_values = np.linspace(0, self.image.shape[0], 3)
                                        if self.alpha<L_alpha_cutoff:
                                                plt.plot([self.rotate_pred[1]] * len(y_values), y_values, "r-") # Left
                                                plt.hlines(y=self.image.shape[0] // 4, xmin=self.rotate_pred[2], xmax=self.rotate_pred[1], colors="red", linestyles="dashed", label="Distance Left")
                                        else:
                                                plt.plot([self.rotate_pred[1]] * len(y_values), y_values, "g-") # Left
                                                plt.hlines(y=self.image.shape[0] // 4, xmin=self.rotate_pred[2], xmax=self.rotate_pred[1], colors="green", linestyles="dashed", label="Distance Left")
                                        plt.plot([self.rotate_pred[2]] * len(y_values), y_values, "w-")
                                        #plt.legend()
                                        plt.savefig('result/result.png', bbox_inches='tight', facecolor='black')
                                        #Show
                                        self._show_pixmap('result/result.png')
                                
                                elif inspire_check_status == True and lt_check_status == False and rt_check_status == False:
                                        plt.imshow(self.overlay)
                                        plt.axis('off')
                                        plt.savefig('result/result.png', bbox_inches='tight', facecolor='black')
                                        #Show
                                        self._show_pixmap('result/result.png')
                                        
                                else:
                                        plt.imshow(self.image)
                                        plt.axis('off')
                                        plt.savefig('result/result.png', bbox_inches='tight', facecolor='black')
                                        #Show
                                        self._show_pixmap('result/result.png')
                        
                        except:
                                self.img.setText("Please choose image")
                
                else:
                        try: 
                                plt.imshow(self.image)
                                plt.axis('off')
                                plt.savefig('result/result.png', bbox_inches='tight', facecolor='black')
                                #Show
                                self._show_pixmap('result/result.png') 
                        except:
                                self.img.setText("Please choose image")

        def _show_pixmap(self, source):
                if isinstance(source, str):
                        pix = QtGui.QPixmap(source)
                else:
                        pix = source
                pix = pix.scaled(self.img.width(), self.img.height(),
                                  QtCore.Qt.KeepAspectRatio,
                                  QtCore.Qt.SmoothTransformation)
                self.img.setScaledContents(False)
                self.img.setPixmap(pix)

if __name__ == "__main__":
        import sys
        app = QtWidgets.QApplication(sys.argv)
        MainWindow = QtWidgets.QMainWindow()
        ui = Ui_MainWindow()
        ui.setupUi(MainWindow)
        MainWindow.showFullScreen()
        sys.exit(app.exec_())