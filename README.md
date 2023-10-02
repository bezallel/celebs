# NaijaCelebs

![celeb](https://user-images.githubusercontent.com/59312765/208271630-7c6efa3a-de53-4e44-aaa9-871f9c313660.png)


```python
import numpy as np
import cv2
from matplotlib import pyplot as plt

%matplotlib inline
```


```python
img = cv2.imread('./test/5-24.jpg')
img.shape
```




    (442, 437, 3)




```python
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
gray.shape
```




    (442, 437)




```python
gray
```




    array([[255, 255, 255, ..., 255, 255, 255],
           [255, 255, 255, ..., 255, 255, 255],
           [255, 255, 255, ..., 255, 255, 255],
           ...,
           [255, 255, 255, ..., 255, 255, 255],
           [255, 255, 255, ..., 255, 255, 255],
           [255, 255, 255, ..., 255, 255, 255]], dtype=uint8)




```python
plt.imshow(gray, cmap='gray')
```




    <matplotlib.image.AxesImage at 0x17691321000>



![output_4_1](https://user-images.githubusercontent.com/59312765/208316061-26a0f26b-3aee-4a20-b9c9-e35abaf6b8c3.png)
    



```python
face_cascade = cv2.CascadeClassifier('./opencv/haarcascades/haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('./opencv/haarcascades/haarcascade_eye.xml')

faces = face_cascade.detectMultiScale(gray, 1.3, 5)
faces
```




    array([[178,  27,  82,  82]])




```python
(x,y,w,h) = faces[0]
x,y,w,h
```




    (178, 27, 82, 82)




```python
face_img = cv2.rectangle(img, (x,y), (x+w,y+h), (255,0,0),2)
plt.imshow(face_img)
```




    <matplotlib.image.AxesImage at 0x17691913130>




    
![output_7_1](https://user-images.githubusercontent.com/59312765/208316083-b8f527f7-43c4-427d-bf76-d35f6afb3017.png)
    



```python
cv2.destroyAllWindows()
for (x,y,w,h) in faces:
    face_img = cv2.rectangle(img, (x,y), (x+w, y+h), (255,0,0),2)
    roi_gray = gray[y:y+h, x:x+w]
    roi_color = face_img[y:y+h, x:x+w]
    eyes = eye_cascade.detectMultiScale(roi_gray)
    for (ex,ey,ew,eh) in eyes:
        cv2.rectangle(roi_color,(ex,ey), (ex+ew,ey+eh), (0,255,0),2)
        
        
plt.figure()
plt.imshow(face_img, cmap='gray')
plt.show()
```


    
![output_8_0](https://user-images.githubusercontent.com/59312765/208316095-164a82bf-866a-4d5b-b88a-cd1fc7c40714.png)
    



```python
%matplotlib inline
plt.imshow(roi_color, cmap='gray')
```




    <matplotlib.image.AxesImage at 0x17690ccf0d0>




![output_9_1](https://user-images.githubusercontent.com/59312765/208316105-eebdfacc-807d-4700-9ba2-75ed37e3fba5.png)
    



```python
def get_cropped_image_if_2_eyes(image_path):
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3,5)
    for (x,y,w,h) in faces:
            roi_gray = gray[y:y+h, x:x+w]
            roi_color = img[y:y+h, x:x+w]
            eyes = eye_cascade.detectMultiScale(roi_gray)
            if len(eyes) >= 2:
                return roi_color    
```


```python
original_image = cv2.imread('./test/5-24.jpg')
plt.imshow(original_image)
```




    <matplotlib.image.AxesImage at 0x17690d59870>




![output_11_1](https://user-images.githubusercontent.com/59312765/208316122-f9215ba0-4b88-467a-8aa1-629d7db83caf.png)
    



```python
cropped_image = get_cropped_image_if_2_eyes('./test/5-24.jpg')
plt.imshow(cropped_image)
```




    <matplotlib.image.AxesImage at 0x17690dbb400>




![output_12_1](https://user-images.githubusercontent.com/59312765/208316140-bef9b8e5-40f0-424a-aded-c3ae14f65e80.png)
    



```python
path_to_data = './data/'
path_to_cr_data = './data/cropped/'
```


```python
import os
img_dirs = []
for entry in os.scandir(path_to_data):
    if entry.is_dir():
        img_dirs.append(entry.path)
```


```python
img_dirs
```




    ['./data/Burna Boy',
     './data/Davido',
     './data/Falz',
     './data/Tems',
     './data/Tiwa Savage',
     './data/Yemi Alade']



### Create Cropped Folder


```python
import shutil
if os.path.exists(path_to_cr_data):
    shutil.rmtree(path_to_cr_data)
os.mkdir(path_to_cr_data)
```


```python
cropped_image_dirs = []
celebrity_file_names_dict = {}

for img_dir in img_dirs:
    count = 1
    celebrity_name = img_dir.split('/')[-1]   
    print(celebrity_name)
    
    celebrity_file_names_dict[celebrity_name] = []
    
    for entry in os.scandir(img_dir):
        roi_color = get_cropped_image_if_2_eyes(entry.path)
        if roi_color is not None:
            cropped_folder = path_to_cr_data + celebrity_name
            if not os.path.exists(cropped_folder):
                os.makedirs(cropped_folder)
                cropped_image_dirs.append(cropped_folder)
                print('Creating cropped images in folder:', cropped_folder)
                
                
            cropped_file_name = celebrity_name + str(count) + '.jpg'
            cropped_file_path = cropped_folder + '/' + cropped_file_name
            
            cv2.imwrite(cropped_file_path, roi_color)
            celebrity_file_names_dict[celebrity_name].append(cropped_file_path)
            count += 1                
```

    Burna Boy
    Creating cropped images in folder: ./data/cropped/Burna Boy
    Davido
    Creating cropped images in folder: ./data/cropped/Davido
    Falz
    Creating cropped images in folder: ./data/cropped/Falz
    Tems
    Creating cropped images in folder: ./data/cropped/Tems
    Tiwa Savage
    Creating cropped images in folder: ./data/cropped/Tiwa Savage
    Yemi Alade
    Creating cropped images in folder: ./data/cropped/Yemi Alade
    


```python
import numpy as np 
import matplotlib.pyplot as plt 
import pywt

def w2d(img, mode = 'haar', level=1):
    imArray = img
    
    imArray = cv2.cvtColor(imArray, cv2.COLOR_RGB2GRAY)
    imArray = np.float32(imArray)
    imArray /= 255;
    
    coeffs = pywt.wavedec2(imArray, mode, level=level)
    
    coeffs_H=list(coeffs)
    coeffs_H[0] *= 0;
    
    imArray_H = pywt.waverec2(coeffs_H, mode);
    imArray_H *= 255;
    imArray_H = np.uint8(imArray_H)
    
    return imArray_H    
```


```python
im_har = w2d(cropped_image, 'db1', 5)
plt.imshow(im_har, cmap='gray')
```




    <matplotlib.image.AxesImage at 0x17690e4b1f0>





![output_20_1](https://user-images.githubusercontent.com/59312765/208316149-e9f8be3b-38a6-4569-bb32-d6bf67211545.png)
    



```python
celebrity_file_names_dict
```
![bs6](https://user-images.githubusercontent.com/59312765/208316375-bffbe1ce-78c6-4993-8035-5ba90c41d5e8.png)
```python
class_dict = {}
count = 0
for celebrity_name in celebrity_file_names_dict.keys():
    class_dict[celebrity_name] = count
    count = count + 1
class_dict
```




    {'Burna Boy': 0,
     'Davido': 1,
     'Falz': 2,
     'Tems': 3,
     'Tiwa Savage': 4,
     'Yemi Alade': 5}



### Data Preparation 
#### Splitting the wavelets transformed data into X and Y variables for training.


```python
x = []
y = []

for celebrity_name, training_files in celebrity_file_names_dict.items():
    for training_image in training_files:
        img = cv2.imread(training_image)
        scaled_raw_img = cv2.resize(img, (32, 32))
        img_har = w2d(img, 'db1', 5)
        scaled_img_har = cv2.resize(img_har, (32, 32))
        combined_img = np.vstack((scaled_raw_img.reshape(32*32*3,1),scaled_img_har.reshape(32*32,1)))
        x.append(combined_img)
        y.append(class_dict[celebrity_name])
```


```python
len(x[0])
```




    4096




```python
x = np.array(x).reshape(len(x), 4096).astype(float)
x.shape
```




    (181, 4096)




```python
x[0]
```




    array([135., 135., 135., ...,  68.,   1.,   0.])



### Training the Model.


```python
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
```


```python
x_train, x_test, y_train, y_test = train_test_split(x,y, random_state = 0)

pipe = Pipeline([('scaler', StandardScaler()), ('svc', SVC(kernel = 'rbf', C = 10))])
pipe.fit(x_train, y_train)
pipe.score(x_test, y_test)
```




    0.782608695652174

