## API IMPORTS
import numpy as np
from api.ml_processing import gradCam, image_ct_or_not, image_xray_or_not, lung_segment, predict_ct_scan, predict_xray_for_5_diseases, xray_gradcam
from rest_framework.permissions import AllowAny, IsAuthenticated
from rest_framework.decorators import authentication_classes, permission_classes
from rest_framework import generics, status, permissions, serializers
from django.http import HttpResponse
from rest_framework.response import Response
from django.core.files.storage import FileSystemStorage

from rest_framework.parsers import JSONParser
from rest_framework.renderers import JSONRenderer, TemplateHTMLRenderer

from django.template.context_processors import csrf
from django.views.decorators.csrf import csrf_exempt
from django.shortcuts import render, redirect, HttpResponse
from .models import *
# Add in header
from django.http import JsonResponse
from django.middleware.csrf import get_token

#YoloV5 Detection Libraries by PYTorch
import detect
# from keras import optimizers
# from keras import layers
# from classification_models.tfkeras import Classifiers
# from tensorflow.keras import Model
# import tensorflow as tf

# import utils
import json
import os
from datetime import datetime, timezone
import time
import pytz
import numpy as np


# @permission_classes((AllowAny, )) # Allow any user
# @permission_classes((IsAuthenticated, )) # Allow Authenticated User
# @csrf_exempt
# permission_classes = (permissions.AllowAny,)
# permission_classes = (permissions.IsAuthenticated,)

## API Pattern: # http://127.0.0.1:8000/api/YOLOAPI

# YOLO API:

@permission_classes((AllowAny, ))
class GetYOLODetection(generics.RetrieveAPIView):
    permission_classes = (permissions.AllowAny,)
    def post(self, request):
        name = self.request.data.get('name',None)
        age = self.request.data.get('age',None)
        bloodGroup = self.request.data.get('bloodGroup',None)
        weight = self.request.data.get('weight',None)
        token = self.request.data.get('token',None)
        doctorID = self.request.data.get('doctorID',None)
        gender = self.request.data.get('gender',None)
        image = request.FILES['xray']
        
        ID_millis = int(datetime.now().timestamp() * 1000000)

        OrgImageName = image.name
        obj = FileSystemStorage()       ## Created Object of FileSystemStorage 
        ImagePath = obj.save(OrgImageName, image)
        upload_image_path = obj.save("Image/{0}/".format(ID_millis) + OrgImageName, image) 
        upload_image_path_media = "./media/" + upload_image_path
        # print(ImagePath)
        # print(type(ImagePath))
        rslt = image_xray_or_not(img_path = upload_image_path_media)

        if rslt == False:
            os.remove("./media/"+ImagePath)
            response = {"imageValid":"False"}
            response_json = JSONRenderer().render(response)
            return HttpResponse(response_json, content_type='application/json', status = status.HTTP_400_BAD_REQUEST)
        
        ## Lung Segmentation
        lung_segment_path = lung_segment(upload_image_path_media, ID_millis, OrgImageName)
        resultOfYOLO = detect.detect(upload_image_path_media, save_dir="./media/Image/{0}/".format(ID_millis), user_id=123456, aws_s3_path="./media/aws_s3_bucket/", save_img=False)

        Detections = resultOfYOLO.get('Detections')
        detected_path = resultOfYOLO.get('new_path')      
    
        # with open("."+detected_path, "rb") as image2string:
            # converted_string = base64.b64encode(image2string.read())
        # print(type(converted_string))
        # detected_path = converted_string
        detections_label = ""
        detections_conf = ""
        detections_label_List = []
        detections_conf_List = []

        for string in Detections:
            detections_label  = detections_label + string.rsplit(" ",1)[0] + ", "
            detections_conf  = detections_conf + string.split(" ",-1)[-1] + ", "
            detections_label_List.append(string.rsplit(" ",1)[0])
            detections_conf_List.append(string.split(" ",-1)[-1])

        detections_label = detections_label[0:len(detections_label)-2]
        detections_conf = detections_conf[0:len(detections_conf)-2]
        if len(detections_label) == 0:
            detections_label = "No Disease Found."
            detections_conf_List = "0"
            detections_conf = "0"

        detections_conf_List = [int(float(i)*100) for i in detections_conf_List]


        print("resultOfYOLO.get('new_path')", resultOfYOLO.get('new_path'))
        processedImage_paths = [
            {
                "name": "Lung Segmentation",
                "image_path": lung_segment_path
            },
            {
                "name": "Predicated Image",
                "image_path": "/" + resultOfYOLO.get('new_path').replace("\\", "/")
            }
        ]

        ImageTableObj = ImageTable.objects.create()
        ImageTableObj.DiseaseName = detections_label
        ImageTableObj.Percentage = str(detections_conf_List)
        ImageTableObj.UploadedImage = "/media/"+upload_image_path
        ImageTableObj.ProcessedColoredImage = processedImage_paths[1]["image_path"]
        ImageTableObj.boundingBoxImage = processedImage_paths[0]["image_path"]
        ImageTableObj.save()

        os.remove("./media/"+ImagePath)
        detection = {"patiend_id":"millis","processedImage_paths":processedImage_paths,"imageValid":"True","time":"time_now","tokenValid":'true', 'detections_label':detections_label,'detections_conf_List':detections_conf_List}
        response_json = JSONRenderer().render(detection)
        return HttpResponse(response_json, content_type='application/json', status = status.HTTP_200_OK)

@permission_classes((AllowAny, ))
class GetXRayGradCam(generics.RetrieveAPIView):
    def post(self, request):
        name = self.request.data.get('name',None)
        age = self.request.data.get('age',None)
        bloodGroup = self.request.data.get('bloodGroup',None)
        weight = self.request.data.get('weight',None)
        token = self.request.data.get('token',None)
        doctorID = self.request.data.get('doctorID',None)
        gender = self.request.data.get('gender',None)
        image = request.FILES['gradcam']
        
        OrgImageName = image.name 
        ID_millis = int(datetime.now().timestamp() * 1000000)

        obj = FileSystemStorage()       ## Created Object of FileSystemStorage 
        # django_generated_image_name = obj.save(OrgImageName, image)    ## saving the file in the server
        upload_image_path = obj.save("Image/{0}/".format(ID_millis) + OrgImageName, image)
        upload_image_path = "./media/" + upload_image_path

        rslt = image_xray_or_not(img_path = upload_image_path)
        django_generated_image_name = OrgImageName
        if rslt == False:
            os.remove("./media/"+django_generated_image_name)
            response = {"imageValid":"False"}
            response_json = JSONRenderer().render(response)
            return HttpResponse(response_json, content_type='application/json', status = status.HTTP_400_BAD_REQUEST)

        # path = obj.url(django_generated_image_name)
        # testimage ='.'+path
        
        lung_segment_path = lung_segment(upload_image_path, ID_millis, django_generated_image_name)

        resultImg, res_path, multiClassModel = xray_gradcam(upload_image_path, OrgImageName, ID_millis)

        a = predict_xray_for_5_diseases(upload_image_path, multiClassModel)
        a = a*100  ## Lets equalite all predictions on one stage which can be readable.


        message = ""
        message2 = ""
        result = ""
        colour = ""
        symptoms = ""
        confidence = int(a[0][np.argmax(a)])
        print("Percentage Confidence: " + str(confidence/100))

        if(np.argmax(a) == 0 and a[0][0] >= 40):
            print("Covid")
            message = "AI Diagnosis found signs of"
            message2 = " in chest X-ray."
            result = "Covid"
            colour = "danger"
        elif(np.argmax(a) == 1 and a[0][1] >= 40):
            print("Edema")
            message = "AI Diagnosis found signs of"
            message2 = " in chest X-ray."
            result = "Edema"
            colour = "danger"

        elif(np.argmax(a) == 2 and a[0][2] >= 40):
            print("Normal")
            message = "AI Diagnosis found "
            message2 = " in chest X-ray."
            result = "Normal Condition"
            colour = "success"

        elif(np.argmax(a) == 3 and a[0][3] >= 40):
            print("Pneumonia")    
            message = "AI Diagnosis found signs of"
            message2 = " in chest X-ray."
            result = "Pneumonia"
            colour = "danger"

        elif(np.argmax(a) == 4 and a[0][4] >= 40):
            print("TB")    
            message = "AI Diagnosis found signs of"
            message2 = " in chest X-ray."
            result = "Tuberculosis"
            colour = "danger"

        else:
            print("Model is Confused")
            message = "AI Diagnosis doesn't found any signs of"
            message2 = " in chest X-ray."
            result = "Confused"
            colour = "success"

        print("---------------------------\n Hence: " + str(result))

        processedImage_paths = [
            {
                "name": "Lung Segmentation",
                "image_path": lung_segment_path
            },
            {
                "name": "Predicated Image",
                "image_path": res_path
            }
            ]       

        ImageTableObj = ImageTable.objects.create()
        ImageTableObj.ID = ID_millis
        ImageTableObj.DiseaseName = result
        ImageTableObj.Percentage = str(confidence)
        ImageTableObj.UploadedImage = upload_image_path
        ImageTableObj.ProcessedColoredImage = processedImage_paths[1]["image_path"]
        ImageTableObj.boundingBoxImage = processedImage_paths[0]["image_path"]
        ImageTableObj.save()

        response = {"patiend_id":"millis","processedImage_paths":processedImage_paths,"imageValid":"True","time":"time_now",'detections_label':result, 'detections_conf_List':[confidence]}
        response_json = JSONRenderer().render(response)
        return HttpResponse(response_json, content_type='application/json', status = status.HTTP_200_OK)

@permission_classes((AllowAny, ))
class GetCTSCAN(generics.RetrieveAPIView):
    def post(self, request):
        name = self.request.data.get('name',None)
        age = self.request.data.get('age',None)
        bloodGroup = self.request.data.get('bloodGroup',None)
        weight = self.request.data.get('weight',None)
        gender = self.request.data.get('gender',None)
        token = self.request.data.get('token',None)
        doctorID = self.request.data.get('doctorID',None)
        image = request.FILES['ctscan']
        
        ID_millis = int(datetime.now().timestamp() * 1000000)

        ImageName = image.name 
        obj = FileSystemStorage()       ## Created Object of FileSystemStorage 
        upload_image_path = obj.save("Image/{0}/".format(ID_millis) + ImageName, image)
        upload_image_path = "./media/"+upload_image_path
        
        rslt = image_ct_or_not(img_path = upload_image_path)

        if rslt == False:
            response = {"imageValid":"False"}
            response_json = JSONRenderer().render(response)
            return HttpResponse(response_json, content_type='application/json', status = status.HTTP_400_BAD_REQUEST)

        res_path, probabilityCTcovidNormal = predict_ct_scan(upload_image_path, ImageName, ID_millis)

        message = ""
        message2 = ""
        result = ""
        colour = ""
        symptoms = ""
        
        if probabilityCTcovidNormal[0] > 0.5:
            CTcovidNormalPercentage = (probabilityCTcovidNormal[0]*100) 
            CTcovidNormalPrediction = "COVID"
            result = "COVID with " + str(int(CTcovidNormalPercentage)) + "% Confidence"
            message = "AI Diagnosis found symptoms of"
            message2 = " in chest CT-Scan."
            colour = "danger"
        else:
            CTcovidNormalPercentage = (1-probabilityCTcovidNormal[0]*100) 
            CTcovidNormalPrediction = "NON COVID"
            result = "COVID "
            message = "No symptoms of"
            message2 = "found in chest CT-Scan"
            colour = "warning"

        confidence_pred_int = int(CTcovidNormalPercentage*100)
        if confidence_pred_int > 100:
            confidence_pred_int = confidence_pred_int/100

        ImageTableObj = ImageTable.objects.create()
        ImageTableObj.ID = ID_millis
        ImageTableObj.DiseaseName = CTcovidNormalPrediction
        ImageTableObj.Percentage = str(confidence_pred_int)
        ImageTableObj.UploadedImage = upload_image_path
        ImageTableObj.ProcessedColoredImage = res_path
        ImageTableObj.boundingBoxImage = res_path
        ImageTableObj.save()

        # with open("."+res_path, "rb") as image2string:
            # converted_string = base64.b64encode(image2string.read())

        response = {"imageValid":"True","time":"time_now",'path':res_path,'detected_path':"/media/"+res_path.split('/',2)[2],'detections_label':CTcovidNormalPrediction, 'detections_conf_List':[confidence_pred_int]}
        response_json = JSONRenderer().render(response)
        return HttpResponse(response_json, content_type='application/json', status = status.HTTP_200_OK)
