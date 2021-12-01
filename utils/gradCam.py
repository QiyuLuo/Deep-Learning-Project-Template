from torchsummary import summary
import torch
import torch.nn as nn
import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
import torch
from torchvision.models import resnet18
from torchvision import transforms
from pytorch_grad_cam import GradCAM, ScoreCAM, GradCAMPlusPlus, AblationCAM, XGradCAM, EigenCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
def viz(module, input):
    x = input[0][0]
    #最多显示4张图
    min_num = np.minimum(4, x.size()[0])
    for i in range(min_num):
        plt.subplot(1, 4, i+1)
        plt.imshow(x[i].cpu())
    plt.show()

# 定义获取特征图的函数
def farward_hook(module, input, output):
    fmap_block.append(output)

# 计算grad-cam并可视化
def cam_show_img(img, feature_map, grads, out_dir, out_file='cam.jpg'):
    H, W, _ = img.shape
    cam = np.zeros(feature_map.shape[1:], dtype=np.float32)		# 4
    grads = grads.reshape([grads.shape[0],-1])					# 5
    weights = np.mean(grads, axis=1)							# 6
    for i, w in enumerate(weights):
        cam += w * feature_map[i, :, :]							# 7
    cam = np.maximum(cam, 0)
    cam = cam / cam.max()
    cam = cv2.resize(cam, (W, H))

    heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
    cam_img = 0.3 * heatmap + 0.7 * img

    path_cam_img = os.path.join(out_dir, out_file)
    cv2.imwrite(path_cam_img, cam_img)

def mdnetVis():
    rgb = plt.imread('rgb.jpg')
    rgb = rgb.transpose(2, 0, 1) / 255
    inf = plt.imread('inf.jpg')
    inf = inf.transpose(2, 0, 1) / 255
    x1 = torch.from_numpy(rgb).unsqueeze(0).type(torch.float32)
    x2 = torch.from_numpy(inf).unsqueeze(0).type(torch.float32)
    x = [x1, x2]
    model = resnet18()
    for name, m in model.named_modules():
        # 这里只对卷积层的feature map进行显示
        if isinstance(m, torch.nn.Conv2d):
            m.register_forward_pre_hook(viz)
    model.eval()

    # 提取conv1输出的特征图，保存在fmap_block中
    model.feature_rgb.conv1[-1].register_forward_hook(farward_hook)
    model.feature_inf.conv1[-1].register_forward_hook(farward_hook)
    with torch.no_grad():
        output = model(x)
    rgb, inf = fmap_block[0], fmap_block[1]
    rgb = torch.mean(rgb, dim=1).squeeze(0)
    inf = torch.mean(inf, dim=1).squeeze(0)
    np.savetxt('rgb_conv1.txt', rgb)
    np.savetxt('inf_conv1.txt', inf)

    for name, param in model.named_parameters():
        print(name, param.shape)

def grad():
    rgb = plt.imread('rgb.jpg')
    rgb = rgb.transpose(2, 0, 1) / 255
    inf = plt.imread('inf.jpg')
    inf = inf.transpose(2, 0, 1) / 255
    x1 = torch.from_numpy(rgb).unsqueeze(0).type(torch.float32)
    x2 = torch.from_numpy(inf).unsqueeze(0).type(torch.float32)
    x = [x1, x2]
    model = resnet18()
    target_layer = model.resnet
    cam = GradCAM(model=model, target_layer=target_layer, use_cuda=False)
    # 计算cam
    grayscale_cam = cam(input_tensor=x, target_category=None)
    # 展示热力图并保存
    grayscale_cam = grayscale_cam[0]
    visualization = show_cam_on_image(rgb, grayscale_cam)  # (224, 224, 3)
    cv2.imwrite(f'cam_dog.jpg', visualization)

if __name__ == '__main__':
    # 用于存放特征图
    # fmap_block = list()
    # mdnetVis()
    m = nn.Conv3d(16, 33, (3, 5, 2), stride=(2, 1, 1), padding=(4, 2, 0))
    input = torch.randn(20, 16, 10, 50, 100)
    output = m(input)
    print(output.shape) # torch.Size([20, 33, 8, 50, 99])

