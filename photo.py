import os

from django.http import HttpResponse, JsonResponse, FileResponse
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_POST
from django.conf import settings
from django.core.files.storage import FileSystemStorage
from rest_framework.decorators import api_view
from rest_framework.response import Response
from rest_framework import status
from django.shortcuts import render

from . import model_views # - addition ! -s

@api_view(['POST'])
def upload_photo(request):
    print(type(request))
    print(request.FILES.keys())
    if 'temp_image' in request.FILES:
        photo = request.FILES['temp_image']

        # ????????? ????????? ??????????? ??????
        fs = FileSystemStorage(location=settings.UPLOAD_ROOT)
        filename = fs.save(photo.name, photo)
        file_path = os.path.join(settings.UPLOAD_ROOT, filename) # - addition ! -
        # ?????? URL ??????
        file_url = fs.url(filename)

        model_views.process_uploaded_image(file_path) # - addition ! -
        
        # index.html ???????? ????????
        return Response({'message': 'Image Upload Completed', 'file_url': file_url}, status=status.HTTP_201_CREATED) # 원래 문구 : ????? ???로드 ?????
    #else:
        # return Response({'message': str((request.FILES.keys()))}, status=status.HTTP_400_BAD_REQUEST) # 원래 문구 : ????????? ????????????.

        # return Response({'message': 'failed'}, status=status.HTTP_400_BAD_REQUEST) # 원래 문구 : ????????? ????????????.

@api_view(['DELETE'])
def delete_file(request):
    if request.method == 'DELETE':
        # ???버에??? ?????? ?????? 경로 ?????
        file_path1 = os.path.join(settings.UPLOAD_ROOT, 'temp_image.jpg')
        file_path2 = os.path.join(settings.DOWNLOAD_ROOT, 'blended_image.jpg')

        # ????????? 존재?????? 경우 ??????
        if os.path.exists(file_path1) or os.path.exists(file_path2):
            os.remove(file_path1)
            os.remove(file_path2)
            return JsonResponse({'message': 'success'}, status=204) # 원래 문구 : ?????? ?????? ?????
    else:
        return JsonResponse({'message': 'No image or invalid request method'}, status=400) # 원래 문구 : 
    

# def download_photo(request):
#     # ???미??? ????????? 경로
#     image_path = os.path.join('cookie', 'media', 'temp_image.jpg')
    
#     # ???미??? ????????? ????????? FileResponse?? 반환
#     try:
#         with open(image_path, 'rb') as image_file:
#             return FileResponse(image_file)
#     except FileNotFoundError:
#         return HttpResponse('Image not found', status=404)

    

# @api_view(['DELETE'])
# def delete_photo(request, file_path):
#     try:
#         # ???미??? ????????? 경로?????? ????????? ??????
#         if os.path.exists(file_path):
#             os.remove(file_path)
#             return Response({'message': '???미??? ?????? ?????'}, status=status.HTTP_204_NO_CONTENT)
#         else:
#             return Response({'message': '???미????? 존재????? ????????????.'}, status=status.HTTP_404_NOT_FOUND)
#     except Exception as e:
#         return Response({'message': '???미??? ?????? ??????', 'error': str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
