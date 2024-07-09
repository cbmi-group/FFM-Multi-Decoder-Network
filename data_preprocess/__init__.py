import glob
import os


def generate_train_txt():
    root_path = '/data/ldap_shared/home/'
    txt_save_path = os.path.join('../datasets', "train_mito.txt")
    file_path = '../mito_dataset/train/'
    masks_dir = file_path + 'masks_aug/'
    images_dir = file_path + 'images_aug/'
    img_list = glob.glob(os.path.join(images_dir, "*.tif"))
    img_list.sort()
    with open(txt_save_path, 'w') as f:
        for i, p in enumerate(img_list):
            img_name = os.path.split(p)[-1]
            print("==> Process image: %s." % (img_name))
            f.writelines(root_path + images_dir[2:] + img_name + " " + root_path + masks_dir[2:] + img_name + "\n")


def generate_test_txt():
    root_path = '/data/ldap_shared/home/'
    txt_save_path = os.path.join('../datasets', "test_mito.txt")
    file_path = '../mito_dataset/test/'
    masks_dir = file_path + 'masks/'
    images_dir = file_path + 'images/'

    img_list = glob.glob(os.path.join(images_dir, "*.tif"))
    img_list.sort()
    with open(txt_save_path, 'w') as f:
        for i, p in enumerate(img_list):
            img_name = os.path.split(p)[-1]
            print("==> Process image: %s." % (img_name))
            f.writelines(
                root_path + images_dir[2:] + img_name + " " + root_path + masks_dir[2:] + img_name + "\n")


if __name__ == '__main__':
    generate_train_txt()
