import os, sys, torch, io
sys.path.insert(0, 'D:/Qin/project/python/unet/src')
os.chdir('D:/Qin/project/python/unet')

def main():
    torch.load('save_weights/best_temp.pth', map_location='cpu')
    print('OK')

if __name__ == '__main__':
    main()
