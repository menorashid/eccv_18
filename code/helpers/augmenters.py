import numpy as np

def crop_center(img,cropx,cropy):
    # y,x = img.shape
    y = img.shape[0]
    x = img.shape[1]
    startx = x//2-(cropx//2)
    starty = y//2-(cropy//2)    
    return img[starty:starty+cropy,startx:startx+cropx,:]

def random_crop(img,crop_size):
    assert img.shape[0]==img.shape[1]
    img_size = img.shape[0]
    assert crop_size<=img_size
    min_val = 0
    max_val = img_size-crop_size
    x = np.random.randint(min_val,max_val)
    y = np.random.randint(min_val,max_val)
    return img[y:y+crop_size,x:x+crop_size,:]

def horizontal_flip(im):
    if np.random.random()<0.5:
        im = np.array(im[:,::-1,:])
    return im