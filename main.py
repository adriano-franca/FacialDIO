# Import dependencies
from IPython.display import display, Javascript, Image
from google.colab.output import eval_js
from base64 import b64decode, b64encode
import cv2
import numpy as np
import PIL
import io
import html
import time

# Função para converter JavaScript object em imagem OpenCV
def js_to_image(js_reply):
    image_bytes = b64decode(js_reply.split(',')[1])
    jpg_as_np = np.frombuffer(image_bytes, dtype=np.uint8)
    img = cv2.imdecode(jpg_as_np, flags=1)
    return img

# Função para converter bbox (bounding box) OpenCV em bytes para overlay
def bbox_to_bytes(bbox_array):
    bbox_PIL = PIL.Image.fromarray(bbox_array, 'RGBA')
    iobuf = io.BytesIO()
    bbox_PIL.save(iobuf, format='png')
    bbox_bytes = 'data:image/png;base64,{}'.format((str(b64encode(iobuf.getvalue()), 'utf-8')))
    return bbox_bytes

# Inicializar o modelo de detecção de face Haar Cascade
face_cascade = cv2.CascadeClassifier(cv2.samples.findFile(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'))


# Função para carregar o JavaScript com Promise
def setup_javascript():
    js = Javascript('''
        function setup_javascript() {
           return new Promise(resolve => {
               var video;
               var div = null;
               var stream;
               var captureCanvas;
               var imgElement;
               var labelElement;

               var pendingResolve = null;
               var shutdown = false;

               function removeDom() {
                   if(stream){
                    stream.getVideoTracks()[0].stop();
                   }
                   if(video){
                       video.remove();
                   }
                   if (div){
                       div.remove();
                   }
                   video = null;
                   div = null;
                   stream = null;
                   imgElement = null;
                   captureCanvas = null;
                   labelElement = null;
               }


               function onAnimationFrame() {
                   if (!shutdown) {
                       window.requestAnimationFrame(onAnimationFrame);
                   }
                   if (pendingResolve) {
                       var result = "";
                       if (!shutdown) {
                           captureCanvas.getContext('2d').drawImage(video, 0, 0, 640, 480);
                           result = captureCanvas.toDataURL('image/jpeg', 0.8)
                       }
                       var lp = pendingResolve;
                       pendingResolve = null;
                       lp(result);
                   }
               }

               async function createDom() {
                   if (div !== null) {
                       return stream;
                   }

                   div = document.createElement('div');
                   div.style.border = '2px solid black';
                   div.style.padding = '3px';
                   div.style.width = '100%';
                   div.style.maxWidth = '600px';
                   document.body.appendChild(div);

                   const modelOut = document.createElement('div');
                   modelOut.innerHTML = "<span>Status:</span>";
                   labelElement = document.createElement('span');
                   labelElement.innerText = 'No data';
                   labelElement.style.fontWeight = 'bold';
                   modelOut.appendChild(labelElement);
                   div.appendChild(modelOut);

                   video = document.createElement('video');
                   video.style.display = 'block';
                   video.width = div.clientWidth - 6;
                   video.setAttribute('playsinline', '');
                   video.onclick = () => { shutdown = true; };
                   stream = await navigator.mediaDevices.getUserMedia(
                       {video: { facingMode: "environment"}});
                   div.appendChild(video);

                   imgElement = document.createElement('img');
                   imgElement.style.position = 'absolute';
                   imgElement.style.zIndex = 1;
                   imgElement.onclick = () => { shutdown = true; };
                   div.appendChild(imgElement);

                   const instruction = document.createElement('div');
                   instruction.innerHTML =
                       '<span style="color: red; font-weight: bold;">' +
                       'When finished, click here or on the video to stop this demo</span>';
                   div.appendChild(instruction);
                   instruction.onclick = () => { shutdown = true; };

                   video.srcObject = stream;
                   await video.play();

                   captureCanvas = document.createElement('canvas');
                   captureCanvas.width = 640;
                   captureCanvas.height = 480;
                   window.requestAnimationFrame(onAnimationFrame);

                   return stream;
               }


              async function stream_frame(label, imgData) {
                   if (shutdown) {
                      removeDom();
                      shutdown = false;
                      return '';
                   }

                   var preCreate = Date.now();
                   stream = await createDom();

                   var preShow = Date.now();
                   if (label != "") {
                      labelElement.innerHTML = label;
                   }

                   if (imgData != "") {
                      var videoRect = video.getClientRects()[0];
                      imgElement.style.top = videoRect.top + "px";
                      imgElement.style.left = videoRect.left + "px";
                      imgElement.style.width = videoRect.width + "px";
                      imgElement.style.height = videoRect.height + "px";
                      imgElement.src = imgData;
                   }

                   var preCapture = Date.now();
                   var result = await new Promise(function(resolve, reject) {
                      pendingResolve = resolve;
                   });
                   shutdown = false;

                   return {'create': preShow - preCreate,
                           'show': preCapture - preShow,
                           'capture': Date.now() - preCapture,
                           'img': result};
              }
             window.stream_frame = stream_frame
            resolve("ready");
          });

        }
       setup_javascript();
        ''')
    display(js)



# Função para processar cada frame do vídeo
def video_frame(label, bbox):
    data = eval_js('window.stream_frame("{}", "{}")'.format(label, bbox))
    return data

# Iniciar o stream de vídeo da webcam
print("Setting up javascript...")
js_ready = eval_js('setup_javascript()')  # Aguarda o sinal "ready" do JavaScript
print("Javascript is ready:", js_ready)
# Label para o vídeo
label_html = 'Reconhecimento Facial'

# Inicializa o bbox como vazio
bbox = ''

# Loop principal do vídeo
while True:
    js_reply = video_frame(label_html, bbox)
    if not js_reply:
        break

    # Converte a resposta JS para imagem OpenCV
    img = js_to_image(js_reply["img"])

    # Cria um overlay transparente para os bounding boxes
    bbox_array = np.zeros([480, 640, 4], dtype=np.uint8)

    # Converte a imagem para grayscale para detecção facial
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    # Detecta faces
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    # Desenha bounding boxes nas faces detectadas
    for (x, y, w, h) in faces:
        bbox_array = cv2.rectangle(bbox_array, (x, y), (x+w, y+h), (255, 0, 0), 2)

    # Converte o overlay para bytes para exibir no vídeo
    bbox_array[:, :, 3] = (bbox_array.max(axis=2) > 0).astype(int) * 255
    bbox_bytes = bbox_to_bytes(bbox_array)
    bbox = bbox_bytes