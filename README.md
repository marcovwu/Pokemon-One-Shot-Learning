# pokemon
AutoEncoder (One Shot Learning)

python                    3.6.8
pytorch                   1.2.0
torchvision               0.4.0

1.plus whole train data in ./data/Train/

    [1].must have 925 images
        images name rule: pokemon (1).png 、 pokemon (2).png ...
        
    [2].if not have 925 images
        change dataloader number:
          self.proto_list = touch_find_whole_image_name(proto_path,your_images_number)
          self.data_list = touch_find_whole_image_name(data_path,your_images_number)       
 
2.plus head train data in ./hesd/Train/
    image name rule: pokemon (1).png 、 pokemon (2).png ...

if you need to resume model please put in ./models
    and change resume_name==./models/your_model_name.pkl
    
Starting to train AE.py  and result will produce in train_all folder  

(1) Start gui

![image](https://github.com/marcovwu/pokemon/blob/master/file_image/start.JPG)

(2) Chose test image

![image](https://github.com/marcovwu/pokemon/blob/master/file_image/test_image.JPG)

(3) Chose calculate mode

![image](https://github.com/marcovwu/pokemon/blob/master/file_image/chose.JPG)

(4) Answer

![image](https://github.com/marcovwu/pokemon/blob/master/file_image/answer.JPG)

(5) Result

![image](https://github.com/marcovwu/pokemon/blob/master/file_image/result.JPG)
