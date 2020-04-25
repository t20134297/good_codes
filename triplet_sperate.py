import os.path as osp
import glob
import re
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T

from torch.utils.data import Sampler
from collections import defaultdict
import copy
import numpy as np
import random



class Market1501():  #return self.train = [ (img_path(string),pid(int),camid(int)) ] self.query, self.test they are the same as train
    dataset_dir = 'market1501'
    def __init__(self, root):
        self.dataset_dir = osp.join( root, self.dataset_dir )
        self.train_dir = osp.join( self.dataset_dir, 'bounding_box_train' )
        self.query_dir = osp.join(self.dataset_dir, 'query')
        self.gallery_dir = osp.join(self.dataset_dir,'bounding_box_test')

        self._check_before_run()

        train = self._process_dir(self.train_dir, relabel=True)
        query = self._process_dir(self.query_dir)
        gallery = self._process_dir(self.gallery_dir)

        self.train = train
        self.query = query
        self.gallery = gallery

        self.num_train_pids, self.num_train_imgs, self.num_train_cams = self.get_imagedata_info(self.train)
        self.num_query_pids, self.num_query_imgs, self.num_query_cams = self.get_imagedata_info(self.query)
        self.num_gallery_pids, self.num_gallery_imgs, self.num_gallery_cams = self.get_imagedata_info(self.gallery)



    def _process_dir(self, dir_path, relabel=False):
        img_paths = glob.glob(osp.join(dir_path, '*.jpg'))
        pattern = re.compile(r'([-\d]+)_c(\d)')

        pid_container = set()
        for img_path in img_paths:
            pid, _ = map(int, pattern.search(img_path).groups())
            if pid==-1:
                continue
            pid_container.add(pid)
        pid2label = {pid: label for label, pid in enumerate(pid_container)}

        dataset = []
        for img_path in img_paths:
            pid, camid = map(int, pattern.search(img_path).groups())
            if pid == -1:
                continue
            assert 0 <= pid <= 1501
            assert 1<= camid <= 6
            camid = camid - 1
            if relabel:
                pid = pid2label[pid]
            dataset.append( (img_path, pid, camid) )

        return dataset


    def _check_before_run(self):
        if not osp.exists(self.dataset_dir):
            raise RuntimeError('{} is not available' .format(self.dataset_dir))
        if not osp.exists(self.train_dir):
            raise RuntimeError('{} is not available' .format(self.train_dir))
        if not osp.exists(self.query_dir):
            raise RuntimeError('{} is not available' .format(self.query_dir))
        if not osp.exists(self.gallery_dir):
            raise RuntimeError('{} is not available' .format(self.gallery_dir))


    def get_imagedata_info(self,data):
        pids, cams = [], []
        for _, pid, camid in data:
            pids += [pid]
            cams += [camid]
        pids = set(pids)
        cams = set(cams)
        num_pids = len(pids)
        num_cams = len(cams)
        num_imgs = len(data)

        return num_pids, num_imgs, num_cams

def read_image(img_path):
    got_img = False
    if not osp.exists(img_path):
        raise IOError('{} does not exist'.format(img_path))
    while not got_img:
        try:
            img = Image.open(img_path).convert('RGB')
            got_img = True
        except IOError:
            print('IOError incurred when reading {} , will redo'.format(img_path))
            pass
    return img


class ImageDataset(Dataset):  #return image(tensor)  [ (image,pid,camid,img_path) ]
    def __init__(self, dataset, transform=None):
        self.dataset = dataset
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        img_path, pid, camid = self.dataset[index]
        img = read_image(img_path)

        if self.transform is not None:
            img = self.transform(img)

        return img, pid, camid, img_path


def build_transforms(is_train = True):
    if is_train:
        transform = T.Compose(
            [
                T.Resize( [384,128] ),
                T.RandomHorizontalFlip(p=0.5),
                T.Pad(10),
                T.RandomCrop([384,128]),
                T.ToTensor(),
                T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )
    else:
        transform = T.Compose(
            [
                T.Resize([384, 128]),
                T.ToTensor(),
                T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )
    return transform


class RandomIdentiyiSampler(Sampler):
    def __init__(self, data_source, batch_size, num_instance ):
        self.data_source = data_source
        self.batch_size = batch_size
        self.num_instance = num_instance
        self.num_pids_per_batch = self.batch_size // self.num_instance
        self.index_dic = defaultdict(list)

        for index, pid in enumerate( self.data_source ):
            self.index_dic[pid].append( index )
        self.pids = list( self.index_dic.keys() )

        self.length = 0
        for pid in self.pids:
            idxs = self.index_dic[pid]
            num = len( idxs )
            if num < self.num_instance:
                num = self.num_instance
            self.length += num - num % self.num_instance


    def __iter__(self):
        batch_idxs_dict = defaultdict(list)

        for pid in self.pids:
            idxs = copy.deepcopy( self.index_dic[pid] )
            if len(idxs) < self.num_instance:
                idxs = np.random.choice( idxs, size = self.num_instance, replace=True )
            random.shuffle(idxs)
            batch_idxs = []
            for idx in idxs:
                batch_idxs.append( idx )
                if( len(batch_idxs) == self.num_instance ):
                    batch_idxs_dict[pid].append(batch_idxs)
                    batch_idxs = []

        avai_pids = copy.deepcopy( self.pids )
        final_idxs = []
        while( len(avai_pids)>= self.num_pids_per_batch ):
            selected_pids = random.sample(avai_pids, self.num_pids_per_batch)
            for pid in selected_pids:
                batch_idxs = batch_idxs_dict[pid].pop(0)
                final_idxs.extend(batch_idxs)
                if len(batch_idxs_dict[pid])==0:
                    avai_pids.remove(pid)

        self.lenght = len(final_idxs)
        return iter(final_idxs)

    def __len__(self):
        return self.length




dataset = Market1501('/home/ansheng/Desktop/reid_base_market')
t = build_transforms()
img_dataset = ImageDataset(dataset.train,t)
train_loader = DataLoader(img_dataset, batch_size = 15, sampler=RandomIdentiyiSampler(dataset.train,15, 3))

for data in train_loader:
    imgs, pid, camid, img_path = data
    print(pid)
    # print(type(imgs))
    # print(imgs.shape)
    # print(type(pid))
    # print(pid.shape)
    # print(type(camid))
    # print(camid.shape)
    # print(type(img_path))
    break


# path = './imgs/0001_c1s1_001051_00.jpg'
# img =  read_image(path)
# t = build_transforms()
# torch_img = t(img)
# print(img)
# print(type(img))
# print(torch_img)
# print(type(torch_img))
# print(torch_img.shape)







