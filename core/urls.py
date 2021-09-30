from django.urls import path, include
from api import api
import api
from django.contrib import admin    # For admin panel

from django.conf import settings                # These are only for development purpose
from django.conf.urls.static import static      # As Django doen't have proper server
                                                # to serve media files. For Deployment we use Servers.

urlpatterns = [
    path('admin/', admin.site.urls),
    path('api-auth/', include('rest_framework.urls')),
    path('api/', include('api.urls')),

] + static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)
