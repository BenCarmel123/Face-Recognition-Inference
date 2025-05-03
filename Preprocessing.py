import os
from data.export_train_label import create_label1
# from data.image_downloader import download_img
from data.image_downloader_new import download_img

if __name__ == '__main__':
    if not os.path.exists('data/trainCrop') and not os.path.exists('data/train'):
        os.makedirs('data/train', exist_ok=True)
        os.makedirs('data/trainCrop', exist_ok=True)
        download_img()
    create_label1(False)  
    create_label1(True)