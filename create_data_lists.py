from utils import create_data_lists

if __name__ == '__main__':
    create_data_lists(voc07_path='/mnt2/datasets/VOCdevkit/VOC2007', # specify your data root
                      voc12_path='/mnt2/datasets/VOCdevkit/VOC2012',
                      output_folder='./')
