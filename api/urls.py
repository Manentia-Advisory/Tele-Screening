from django.urls import path
from django.conf.urls import url
from . import api
from django.urls import include, path, re_path
from django.conf import settings                # These are only for development purpose
from django.conf.urls.static import static      # As Django doen't have proper server
                                                # to serve media files. For Deployment we use Servers.

urlpatterns = [
    path('getYoloXRay', api.GetYOLODetection.as_view()),
    path('getXRayGradCam', api.GetXRayGradCam.as_view()), 
    path('getCTSCAN', api.GetCTSCAN.as_view())
    
]
if settings.DEBUG:
    urlpatterns + static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)