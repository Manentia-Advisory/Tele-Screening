from django.db import models
from datetime import datetime
from PIL import Image

def processedColoredImageName(instance, filename):
    return 'Image/{0}/processedColored.jpeg'.format(instance.ID)

def UploadedImageName(instance, filename):
    print(instance.ID)
    image_path = 'Image/{0}/uploadedImage.jpeg'.format(instance.ID)
    return image_path

def BoundingBoxImageName(instance, filename):
    return 'Image/{0}/boundingBox.jpeg'.format(instance.ID)

def ContactImage(instance, filename):
    return 'ContactImage/{0}.jpeg'.format(instance.ID)

class ImageTable(models.Model):
    ID = models.IntegerField(default = int(datetime.now().timestamp() * 1000000), primary_key=True)
    CreatedAt = models.DateTimeField(auto_now=True)
    DiseaseName = models.CharField(max_length = 512)
    Percentage = models.CharField(max_length = 128)
    UploadedImage = models.ImageField(upload_to = UploadedImageName)
    ProcessedColoredImage = models.ImageField(upload_to = processedColoredImageName)
    boundingBoxImage = models.ImageField(upload_to = BoundingBoxImageName)
    Remarks = models.CharField(max_length = 512, blank=True)

    # def save(self, *args, **kwargs):
    #     super().save(*args, **kwargs)
    #     print(" ------self.uploadedImage")
    #     print(self.UploadedImage)   # printing blank space
    #     upload_img = Image.open(self.UploadedImage)
    #     upload_img.save(self.UploadedImage.path, quality=100)

class ContactUs(models.Model):
    ID = models.IntegerField(default = int(datetime.now().timestamp() * 1000000), primary_key = True)
    CreatedAt = models.DateTimeField(auto_now = True)
    Email = models.EmailField()
    Subject = models.CharField(max_length = 128)
    Image = models.ImageField(upload_to = ContactImage, blank = True)
    Bug_FeedBack_Query = models.CharField(max_length = 512)