import cv2
import flask
import pytesseract
from PIL import Image
from io import BytesIO
from flask import Flask, request, jsonify, render_template,Response
import matplotlib.pyplot as plt
import numpy as np
import requests
import os
import json
import boto3
import base64
from pytesseract import Output
# bucket_name = 'sagemaker-ap-south-1-096404013368'
# pytesseract.pytesseract.tesseract_cmd = "C:/Program Files/Tesseract-OCR/tesseract.exe"

app = Flask(__name__)
s3_client = boto3.client('s3')
# app.config['JSON_AS_ASCII'] = False

# class MyEncoder(json.JSONEncoder):
#     def default(self, obj):
#         if isinstance(obj, np.integer):
#             return int(obj)
#         elif isinstance(obj, np.floating):
#             return float(obj)
#         elif isinstance(obj, np.ndarray):
#             return obj.tolist()
#         else:
#             return super(MyEncoder, self).default(obj)

# class OCR:
#     def __init__(self):
#         self.preprocessing = Preprocessing()

#     def get_image(self, image_path):
#         img = cv2.imread(image_path)
#         preprocessed_img = self.preprocessing.thresholding(img)
#         text = self.extract_data(preprocessed_img)
#         return text

#     def get_confidence(self, t):
#         co = []
#         # Convert dataframe to string
#         # t = t.to_string()
#         # lines = str(t).splitlines()
#         for i, line in enumerate(t.splitlines()):
#             if i == 0:
#                 pass
#             else:
#                 # Split the line into fields
#                 fields = line.split('\t')

#                 # Extract the text and confidence score
#                 text = fields[-1]
#                 confidence = float(fields[-2])
#                 co.append(confidence)
#                 # Print the text and confidence score
#                 # print(f'Text: {text}, Confidence: {confidence}')
#         print("Overall confidence score", np.mean(co))

#     def extract_data(self, image):
#         t = pytesseract.image_to_data(image, lang='hin')
#         conf = self.get_confidence(t)
#         return pytesseract.image_to_string(image, lang='hin', config='--oem 1 --psm 3')


# class Preprocessing:
#     def __init__(self):
#         pass

#     # get grayscale image
#     def get_grayscale(self, image):
#         return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

#     # noise removal
#     def remove_noise(self, image):
#         return cv2.medianBlur(image, 5)

#     # thresholding
#     def thresholding(self, image):
#         return cv2.threshold(self.get_grayscale(image), 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]

#     # dilation
#     def dilate(self, image):
#         kernel = np.ones((5, 5), np.uint8)
#         return cv2.dilate(image, kernel, iterations=1)

#     # erosion
#     def erode(self, image):
#         kernel = np.ones((5, 5), np.uint8)
#         return cv2.erode(image, kernel, iterations=1)

#     # opening - erosion followed by dilation
#     def opening(self, image):
#         kernel = np.ones((5, 5), np.uint8)
#         return cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)

#     # canny edge detection
#     def canny(self, image):
#         return cv2.Canny(image, 100, 200)

#     # template matching
#     def match_template(self, image, template):
#         return cv2.matchTemplate(image, template, cv2.TM_CCOEFF_NORMED)

#     def plot(self, image, cmap=None):
#         plt.figure(figsize=(3, 3))
#         plt.imshow(image, cmap="gray")


# ocr = OCR()


# @app.route('/')
# def upload_form():
#     return render_template('index.html')




def ocr(image_payload):
    photo = base64.b64decode(image_payload['image_byte'])
    word_file = image_payload["word_data"]
    
    file_bytes = base64.b64decode(word_file.encode('utf-8'))
    with open('word.txt', 'w') as file:
        file.write(file_bytes.decode('utf-8'))
    psm_v = image_payload['psm_v']
    l = image_payload["lan"]
    # 
    nparr = np.frombuffer(photo, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
    
    print("PSM _value is --" , psm_v)
    print("language is---" ,  l)
    text = pytesseract.image_to_string(thresh , lang = f"{l}",config=f'--psm {psm_v} --oem 1 --user-words word.txt')#
    # print(text)
    ###################
    d = pytesseract.image_to_data(image, output_type=Output.DICT,lang= f"{l}")

    fn_image = "dummy_image"
    text_box_res = []
    for i in range(len(d['text'])):
        # txt = d['text'][i]
        # if "\"" == txt:
        #     txt = \
        result = {
            "text": d['text'][i],
            "left": d['left'][i],
            "top": d['top'][i],
            "width": d['width'][i],
            "height": d['height'][i],
            "conf": d['conf'][i]
        }
        if result["text"].strip() == "":
            continue
        else:
            
            if result['conf'] > 35:
                
            
                (x, y, w, h,c) = (result["left"], result["top"], result["width"], result["height"], result["conf"])
                cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
                
        #         print(result)
                # tx = str(result)
                # with open(f"{fn_image}_ocr-output.txt", "a") as f:
                #     f.write("\n")
                #     f.write(tx+"\n")
                text_box_res.append(result)
    cv2.imwrite(f'{fn_image}_final_bounding-box.jpg', image)
    
    pdf_i = pytesseract.image_to_pdf_or_hocr(f'{fn_image}_final_bounding-box.jpg', extension='pdf',lang=f"{l}")
    with open(f'{fn_image}_final_bounding-box.pdf', 'w+b') as f:
        f.write(pdf_i) # pdf type is bytes by default
        
    
    with open(f'{fn_image}_final_bounding-box.pdf', 'rb') as file:
        pdf_bytes = file.read()

    pdf_base64 = base64.b64encode(pdf_bytes).decode('utf-8')

    
    ########### bb box text file update #############
    text_box_res = json.dumps(text_box_res)
    # text_box_res = str(text_box_res)
    with open(f"{fn_image}_ocr-output.txt","w", encoding="utf-8") as ff:
            ff.write(text_box_res)

    with open(f"{fn_image}_ocr-output.txt", 'rb') as file1:
        file_contents_txt = file1.read().decode('utf-8')
    ############################
    with open(f'{fn_image}_final_bounding-box.jpg', 'rb') as file2:
        image_bytes = file2.read()
        file_contents_bbox = base64.b64encode(image_bytes).decode('utf-8')
    # return file_contents
    ##################
    return text,file_contents_txt,file_contents_bbox ,pdf_base64

@app.route('/ping',methods=['GET'])
def ping():
    return Response(response="OK", status=200)




@app.route('/invocations', methods=['POST'])
def transformation():
  
    flask.request.content_type == 'application/json'
    data = flask.request.data.decode('utf-8')
    data = json.loads(data)

    print(data)
    
    ## s3:bucket/ocr-image/1.png
    
    
    bucket = data['bucket']
    image_uri = data['image_uri']
    word_uri = data['word_uri']
    psm_v = data['psm_v']
    l = data['lang']
    
    
    # bucket = 'sagemaker-ap-south-1-087518342417'
    # image_uri  = 'ocr-demo/abc.png'

    download_file_name = image_uri.split('/')[-1]
    word_download_file_name = word_uri.split('/')[-1]
    print ("<<<<download_file_name ", download_file_name)
    s3_client.download_file(bucket, image_uri, download_file_name)
    s3_client.download_file(bucket, word_uri, word_download_file_name)

    print('Download finished!')
    
    print('Start to inference:')

    # image = Image.open(BytesIO(response.read()))
    # preprocessed_img = ocr.preprocessing.thresholding(np.array(image))

    # text = ocr.extract_data(preprocessed_img)

    # payload=json.dumps(text,ensure_ascii=False)

    # print(text)

    # return jsonify({'text': text})
    
    # psm_v = 3
    # l = "hin"
    with open(download_file_name, 'rb') as image:
        img_bytes = image.read()

    
    with open(word_download_file_name, 'rb') as file:
       file_bytes = file.read()
    # files = {"image_byte": base64.b64encode(img_bytes).decode(),"psm_v":psm_v,"lan" : l}
    # js = json.dumps(files)

    file_data = {
        'word_data': base64.b64encode(file_bytes).decode(),
        "image_byte": base64.b64encode(img_bytes).decode(),
        "psm_v":psm_v,
        "lan" : l}
    js = json.dumps(file_data)
    
    # json_data = json.dumps(file_data)



    # results = ocr(json.loads(js))
    results,file_contents_txt,file_contents_bbox,pdf_base64 = ocr(json.loads(js))
    print(results)
    
    
    # with open(download_file_name, 'rb') as image:
    #     img_bytes = image.read()
    #     files = {"image_byte": base64.b64encode(img_bytes).decode()}
    #     js = json.dumps(files)
    

    # results = ocr(json.loads(js))

    inference_result = {
        'output': results,
        'file_contents_txt' : file_contents_txt,
        'file_contents_bbox' : file_contents_bbox,
        'pdf_base64' : pdf_base64
        }
    


    resultjson = json.dumps(inference_result)


    return flask.Response(response=resultjson, status=200, mimetype='application/json')


# if __name__ == '__main__':
#     app.run(debug = True)
