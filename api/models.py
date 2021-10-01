from django.db import models
from datetime import datetime

def processedColoredImageName(instance, filename):
    return 'Image/{0}/processedColored.jpeg'.format(instance.ID)

def UploadedImageName(instance, filename):
    return 'Image/{0}/uploadedImage.jpeg'.format(instance.ID)

def BoundingBoxImageName(instance, filename):
    return 'Image/{0}/boundingBox.jpeg'.format(instance.ID)

def ContactImage(instance, filename):
    return 'ContactImage/{0}.jpeg'.format(instance.ID)

class ImageTable(models.Model):
    ID = models.IntegerField(default = int(datetime.now().timestamp() * 1000000), primary_key=True, editable=False)
    CreatedAt = models.DateTimeField(auto_now=True)
    DiseaseName = models.CharField(max_length = 512)
    Percentage = models.CharField(max_length = 128)
    UploadedImage = models.ImageField(upload_to = UploadedImageName)
    ProcessedColoredImage = models.ImageField(upload_to = processedColoredImageName)
    boundingBoxImage = models.ImageField(upload_to = BoundingBoxImageName)
    Remarks = models.CharField(max_length = 512, blank=True)

class ContactUs(models.Model):
    ID = models.IntegerField(default = int(datetime.now().timestamp() * 1000000), primary_key = True, editable=False)
    CreatedAt = models.DateTimeField(auto_now = True)
    Email = models.EmailField()
    Subject = models.CharField(max_length = 128)
    Image = models.ImageField(upload_to = ContactImage, blank = True)
    Bug_FeedBack_Query = models.CharField(max_length = 512)