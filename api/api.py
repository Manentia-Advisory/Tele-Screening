## API IMPORTS
from api.ml_processing import image_xray_or_not
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

# Add in header
from django.http import JsonResponse
from django.middleware.csrf import get_token

#YoloV5 Detection Libraries by PYTorch
import detect

# import utils
import json
import os


# @permission_classes((AllowAny, )) # Allow any user
# @permission_classes((IsAuthenticated, )) # Allow Authenticated User
# @csrf_exempt
# permission_classes = (permissions.AllowAny,)
# permission_classes = (permissions.IsAuthenticated,)

## API Pattern: # http://127.0.0.1:8000/api/YOLOAPI

# YOLO API:
class GetYOLODetection(generics.RetrieveAPIView):
    def post(self, request):
        image = request.FILES['xray']
        ImageName = image.name 

        obj = FileSystemStorage()       ## Created Object of FileSystemStorage 
        ImagePath = obj.save(ImageName, image) 

        result = image_xray_or_not(img_path = "./media/"+ImagePath)

        if result == False:
            os.remove("./media/"+ImagePath)
            response = {"imageValid":"False"}
            response_json = JSONRenderer().render(response)
            return HttpResponse(response_json, content_type='application/json')
        
        resultOfYOLO = detect.detect(ImagePath, save_dir="./media/X_Ray_Detections/", user_id=123456, aws_s3_path="./media/aws_s3_bucket/", save_img=False)

        Detections = resultOfYOLO.get('Detections')
        detected_path = resultOfYOLO.get('new_path')
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
        print(detections_conf_List)
        response_json = JSONRenderer().render(detections_label)
        return HttpResponse(response_json, content_type='application/json')

class GetXRayGradCam(generics.RetrieveAPIView):
    def post(self, request):
        image = request.FILES['gradcam']
        ImageName = image.name 

        obj = FileSystemStorage()       ## Created Object of FileSystemStorage 
        ImagePath = obj.save(ImageName, image)    ## saving the file in the server
        result = image_xray_or_not(img_path = "./media/"+ImagePath)
        if result == False:
            os.remove("./media/"+ImagePath)
            response = {"imageValid":"False"}
            response_json = JSONRenderer().render(response)
            return HttpResponse(response_json, content_type='application/json')

        temp = 'xray_img/' + ImageName
        temp1 = 'xray_img/' + ImagePath
        temp2 = 'media/' + ImageName
        path = obj.url(ImagePath)
        testimage ='.'+path

        # covidNormal = load_model('model/CovidNormal.h5')
        # PenumoniaNormal = load_model('model/PneumoniaNormal.h5')
        # CovidPneumonia = load_model('model/CovidPneumonia.h5')
        # multiClassModel = load_model('model/customTLWeights_NEW.h5')

        # covidNormal = load_model('model/CovidNormal.h5')
        # PenumoniaNormal = load_model('model/PneumoniaNormal.h5')

        nb_classes = 5   # number of classes
        img_width, img_height = 256, 256  # change based on the shape/structure of your images
        img_size = 256
        learn_rate = 0.0001  # sgd learning rate
        seresnet152, _ = Classifiers.get('seresnet152')
        base = seresnet152(input_shape=(img_size, img_size, 3), include_top=False, weights='imagenet')
        x = base.output
        x = layers.GlobalAveragePooling2D()(layers.Dropout(0.16)(x))
        x = layers.Dropout(0.3)(x)
        preds = layers.Dense(nb_classes, 'sigmoid')(x)
        multiClassModel=Model(inputs=base.input,outputs=preds)
        loss= tf.keras.losses.BinaryCrossentropy(label_smoothing=0.0)
        multiClassModel.compile(optimizers.Adam(lr=learn_rate),loss=loss,metrics=[tf.keras.metrics.AUC(multi_label=True)])

        multiClassModel.load_weights('model/customTLWeights_NEW_WITH_TB.h5')
        
        resultImg = gradCam(temp2, multiClassModel) 
        res_path = "media/XRay_GradCam/"
        res_path1 = "/media/XRay_GradCam/" + ImageName
        res_path += ImageName
        print(res_path)
        resultImg.save(res_path)                # result Image Path
        res_path = "/" + res_path 
        
        # CovidPneumonia = load_model('model/CovidPneumonia.h5')

        # image = cv2.imread(testimage) # read file 
        # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) # arrange format as per keras
        # image = cv2.resize(image,(224,224))
        # image = np.array(image) / 255
        # image = np.expand_dims(image, axis=0)

        x = load_img(testimage, target_size=(256,256))
        x = img_to_array(x)
        x = np.expand_dims(x, axis=0)
        a = multiClassModel.predict(x)
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

        os.remove(testimage)
        path = '/media/' + temp
        print("---------------------------\n Hence: " + str(result))

        mail = list(DoctorDetails.objects.filter(DoctorID=doctorID).values('Email'))[0].get('Email')
        add_data = X_Ray_Image.objects.create()
        d = Doctor_Profile.objects.get(Email = mail)
        add_data.obj_id = millis
        add_data.Doctor_Name = d.Doctor_Name
        add_data.Hospital_Name = d.Hospital_Name
        add_data.Hospital_Address = d.Hospital_Address
        d.total_image_count += 1
        d.xray_image_count += 1
        add_data.Email = mail
        add_data.Image = request.FILES['gradcam']
        add_data.result_image = "/"+res_path1.split('/',2)[2]
        add_data.Patient_age = age
        add_data.Patient_weight = weight
        add_data.Patient_Name = name
        add_data.Patient_gender = gender
        add_data.abnormalities = result
        add_data.Patient_BloodGroup = bloodGroup
        add_data.confidence = str(confidence/100)

        add_data.save()
        d.save()


        with open("."+res_path, "rb") as image2string:
            converted_string = base64.b64encode(image2string.read())

        response = {"imageValid":"True","time":time_now,"doctorName":d.Doctor_Name,"hospitalName":d.Hospital_Name,'path':res_path,'detected_path':converted_string,'detections_label':result, 'detections_conf_List':[confidence]}
        response_json = JSONRenderer().render(response)
        return HttpResponse(response_json, content_type='application/json')

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
        doctorDetails = list(DoctorAuth.objects.filter(DoctorID=doctorID).values('token'))[0].get('token')

        tz = pytz.timezone('Asia/Kolkata')                     # Specify the timezone
        time_now = datetime.now(timezone.utc).astimezone(tz)   # Get current UTC time and 
                                                            # convert it into specified time zone 
        millis = int(time.mktime(time_now.timetuple()))        # convert the time into millisecond


        if token is None:
            response_json = {}
            response_json['tokenValid'] = 'Failed'
            response_json['message'] = 'Authentication Failed, Please Login Again'
            response_json['isRegistered'] = "False"
            response_json = JSONRenderer().render(response_json)
            return HttpResponse(response_json, content_type='application/json')
        
        elif token != doctorDetails:
            response_json = {}
            response_json['tokenValid'] = 'Failed'
            response_json['message'] = 'Authentication Failed, Please Login Again'
            response_json['isRegistered'] = "True"
            response_json = JSONRenderer().render(response_json)
            return HttpResponse(response_json, content_type='application/json')

        else:
            ImageName = image.name 
            obj = FileSystemStorage()       ## Created Object of FileSystemStorage 
            ImagePath = obj.save(ImageName, image)    ## saving the file in the server

            classifier = keras.models.load_model("./model/ctscan_not.h5")
            img_pred = keras.preprocessing.image.load_img("./media/"+ImagePath, target_size = (64, 64))
            img_pred = keras.preprocessing.image.img_to_array(img_pred)
            img_pred = np.expand_dims(img_pred, axis = 0)
            rslt = classifier.predict(img_pred)

            if rslt[0][0] == 1:
                os.remove("./media/"+ImagePath)
                response = {"imageValid":"False"}
                response_json = JSONRenderer().render(response)
                return HttpResponse(response_json, content_type='application/json')

            temp = 'ctscan_img/' + ImageName
            temp1 = 'ctscan_img/' + ImagePath
            temp2 = 'media/' + ImageName
            path = obj.url(ImagePath)
            testimage ='.'+path

            CTcovidNormal = load_model('model/ctscan_VGG16.h5')

            image = cv2.imread(testimage) # read file 
            print("gradCam")
            
            resultImg = CTgradCam(temp2, CTcovidNormal)
            res_path = "media/CTScan_GradCam/"
            res_path += ImageName
            print(res_path)
            resultImg.save(res_path)
            res_path = "/" + res_path

            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) # arrange format as per keras
            image = cv2.resize(image,(224,224))
            image = np.array(image) / 255
            image = np.expand_dims(image, axis=0)
    
            CTcovidNormal = CTcovidNormal.predict(image)
            print(CTcovidNormal)
            probabilityCTcovidNormal = CTcovidNormal[0]

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
            

            mail = list(DoctorDetails.objects.filter(DoctorID=doctorID).values('Email'))[0].get('Email')
            add_data = CT_Scan_Image.objects.create()
            d = Doctor_Profile.objects.get(Email = mail)
            confidence_pred = [str(confidence_pred_int)]
            add_data.obj_id = millis
            add_data.Doctor_Name = d.Doctor_Name
            add_data.Hospital_Name = d.Hospital_Name
            add_data.Hospital_Address = d.Hospital_Address
            d.total_image_count += 1
            d.ctscan_image_count += 1
            add_data.Email = mail
            add_data.Patient_age = age
            add_data.Patient_weight = weight
            add_data.Patient_Name = name
            add_data.Patient_gender = gender
            add_data.Image = request.FILES['ctscan']
            add_data.result_image = "/"+res_path.split('/',2)[2]
            add_data.COVID_Prediction = CTcovidNormalPrediction
            add_data.Patient_BloodGroup = bloodGroup
            add_data.confidence = confidence_pred[0]
            add_data.save()
            d.save()


            with open("."+res_path, "rb") as image2string:
                converted_string = base64.b64encode(image2string.read())

            response = {"imageValid":"True","time":time_now,"doctorName":d.Doctor_Name,"hospitalName":d.Hospital_Name,'path':res_path,'detected_path':converted_string,'detections_label':CTcovidNormalPrediction, 'detections_conf_List':[confidence_pred_int]}
            response_json = JSONRenderer().render(response)
            return HttpResponse(response_json, content_type='application/json')
