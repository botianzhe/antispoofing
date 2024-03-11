import dlib
from skimage import io
from skimage import transform as sktransform
import numpy as np
from matplotlib import pyplot as plt
import json
import os
import random
from PIL import Image
from imgaug import augmenters as iaa
from faker.DeepFakeMask import dfl_full,facehull,components,extended
import cv2

def name_resolve(path):
    # print('resolve',path)
    vid_id, frame_id = path.split('/')[-2:]
    return vid_id, frame_id[:-4] 
    
def total_euclidean_distance(a,b):
    assert len(a.shape) == 2
    return np.sum(np.linalg.norm(a-b,axis=1))

def random_get_hull(landmark,img1):
    mask = components(landmarks=landmark.astype('int32'),face=img1, channels=3).mask
    return mask/255


def random_erode_dilate(mask, ksize=None):
    if random.random()>0.5:
        if ksize is  None:
            ksize = random.randint(1,21)
        if ksize % 2 == 0:
            ksize += 1
        mask = np.array(mask).astype(np.uint8)*255
        kernel = np.ones((ksize,ksize),np.uint8)
        mask = cv2.erode(mask,kernel,1)/255
    else:
        if ksize is  None:
            ksize = random.randint(1,5)
        if ksize % 2 == 0:
            ksize += 1
        mask = np.array(mask).astype(np.uint8)*255
        kernel = np.ones((ksize,ksize),np.uint8)
        mask = cv2.dilate(mask,kernel,1)/255
    return mask


# borrow from https://github.com/MarekKowalski/FaceSwap
def blendImages(src, dst, mask, featherAmount=0.2):
   
    maskIndices = np.where(mask != 0)
    
    src_mask = np.ones_like(mask)
    dst_mask = np.zeros_like(mask)

    maskPts = np.hstack((maskIndices[1][:, np.newaxis], maskIndices[0][:, np.newaxis]))
    faceSize = np.max(maskPts, axis=0) - np.min(maskPts, axis=0)
    featherAmount = featherAmount * np.max(faceSize)

    hull = cv2.convexHull(maskPts)
    dists = np.zeros(maskPts.shape[0])
    for i in range(maskPts.shape[0]):
        # print(hull.shape,maskPts[i, 0], maskPts[i, 1])
        dists[i] = cv2.pointPolygonTest(hull, (int(maskPts[i, 0]), int(maskPts[i, 1])), True)

    weights = np.clip(dists / featherAmount, 0, 1)

    composedImg = np.copy(dst)
    composedImg[maskIndices[0], maskIndices[1]] = weights[:, np.newaxis] * src[maskIndices[0], maskIndices[1]] + (1 - weights[:, np.newaxis]) * dst[maskIndices[0], maskIndices[1]]

    composedMask = np.copy(dst_mask)
    composedMask[maskIndices[0], maskIndices[1]] = 255
    return composedImg, composedMask

def colorTransfer(src, dst, mask):
    transferredDst = np.copy(dst)
    
    maskIndices = np.where(mask != 0)
    

    maskedSrc = src[maskIndices[0], maskIndices[1]].astype(np.int32)
    maskedDst = dst[maskIndices[0], maskIndices[1]].astype(np.int32)

    meanSrc = np.mean(maskedSrc, axis=0)
    meanDst = np.mean(maskedDst, axis=0)

    maskedDst = maskedDst - meanDst
    maskedDst = maskedDst + meanSrc
    maskedDst = np.clip(maskedDst, 0, 255)

    transferredDst[maskIndices[0], maskIndices[1]] = maskedDst

    return transferredDst


        
def get_blended_face(background_face_path,landmarks_record,data_list):
    background_face = io.imread(background_face_path)
    background_landmark = landmarks_record[background_face_path]
    
    foreground_face_path = search_similar_face(background_landmark,background_face_path,data_list,landmarks_record)
    # print(foreground_face_path)
    foreground_face = io.imread(foreground_face_path)
    # down sample before blending
    aug_size = random.randint(128,256)
    background_landmark = background_landmark * (aug_size/256)
    foreground_face = sktransform.resize(foreground_face,(aug_size,aug_size),preserve_range=True).astype(np.uint8)
    background_face = sktransform.resize(background_face,(aug_size,aug_size),preserve_range=True).astype(np.uint8)
    distortion = iaa.Sequential([iaa.PiecewiseAffine(scale=(0.01, 0.15))])
    # get random type of initial blending mask
    mask = random_get_hull(background_landmark, background_face)
    
    #  random deform mask
    mask = distortion.augment_image(mask)
    mask = random_erode_dilate(mask)
    
    # filte empty mask after deformation
    if np.sum(mask) == 0 :
        raise NotImplementedError

    # apply color transfer
    foreground_face = colorTransfer(background_face, foreground_face, mask*255)
    
    # blend two face
    blended_face, mask = blendImages(foreground_face, background_face, mask*255)
    blended_face = blended_face.astype(np.uint8)
    
    # resize back to default resolution
    blended_face = sktransform.resize(blended_face,(256,256),preserve_range=True).astype(np.uint8)
    mask = sktransform.resize(mask,(256,256),preserve_range=True)
    mask = mask[:,:,0:1]
    return blended_face,mask

def search_similar_face(this_landmark,background_face_path,data_list,landmarks_record):
    vid_id, frame_id = name_resolve(background_face_path)
    min_dist = 99999999
    
    # random sample 5000 frame from all frames:
    all_candidate_path = random.sample(data_list, k=1000) 
    
    # filter all frame that comes from the same video as background face
    all_candidate_path = filter(lambda k:name_resolve(k)[0] != vid_id, all_candidate_path)
    all_candidate_path = list(all_candidate_path)
    
    # loop throungh all candidates frame to get best match
    for candidate_path in all_candidate_path:
        # print('cand',candidate_path)
        candidate_landmark = landmarks_record[candidate_path].astype(np.float32)
        candidate_distance = total_euclidean_distance(candidate_landmark, this_landmark)
        if candidate_distance < min_dist:
            min_dist = candidate_distance
            min_path = candidate_path

    return min_path

def gen_one_datapoint(background_face_path,data_list,landmarks_record):
    data_type = random.randint(0,1) 
    if data_type == 1 :
        face_img,mask =  get_blended_face(background_face_path,landmarks_record,data_list)
        # mask = ( 1 - mask ) * mask * 4
        data_type=[0,1]
    else:
        face_img = io.imread(background_face_path)
        face_img = sktransform.resize(face_img,(256,256),preserve_range=True).astype(np.uint8)
        mask = np.zeros((256, 256, 1))
        data_type=[1,0]
    # print('background_face',background_face.shape)
    
    # randomly downsample after BI pipeline
    if random.randint(0,1):
        aug_size = random.randint(64, 256)
        face_img = Image.fromarray(face_img)
        if random.randint(0,1):
            face_img = face_img.resize((aug_size, aug_size), Image.BILINEAR)
        else:
            face_img = face_img.resize((aug_size, aug_size), Image.NEAREST)
        face_img = face_img.resize((256, 256),Image.BILINEAR)
        face_img = np.array(face_img)
        
    # random jpeg compression after BI pipeline
    if random.randint(0,1):
        quality = random.randint(60, 100)
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
        face_img_encode = cv2.imencode('.jpg', face_img, encode_param)[1]
        face_img = cv2.imdecode(face_img_encode, cv2.IMREAD_COLOR)
    
    # face_img = face_img[60:256,30:287,:]
    # mask = mask[60:256,30:287,:]
    
    # random flip
    if random.randint(0,1):
        face_img = np.flip(face_img,1)
        mask = np.flip(mask,1)
        
    return face_img,mask,data_type

def mask_face(background_face,background_landmark):

    mask = random_get_hull(background_landmark, background_face)
    # print(mask.shape)
    # blend two face
    background_face[mask!=0] = 0
    background_face = background_face.astype(np.uint8)
    
    # resize back to default resolution
    background_face = sktransform.resize(background_face,(256,256),preserve_range=True).astype(np.uint8)
    mask = sktransform.resize(mask,(256,256),preserve_range=True)
    mask = mask[:,:,0:1]
    return background_face,mask