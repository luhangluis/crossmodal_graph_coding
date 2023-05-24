import torch
import torchvision
from PIL import Image
import cv2 as cv
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

model = torchvision.models.detection.fasterrcnn_resnet50_fpn(
    pretrained=True,
    weights=torchvision.models.detection.FasterRCNN_ResNet50_FPN_Weights.COCO_V1
)

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
print(device)
model.to(device)
coco_labels_name = ["unlabeled", "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck",
                    "boat",
                    "traffic light", "fire hydrant", "street sign", "stop sign", "parking meter", "bench", "bird",
                    "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "hat",
                    "backpack",
                    "umbrella", "shoe", "eye glasses", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard",
                    "sports_ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard",
                    "tennis racket",
                    "bottle", "plate", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
                    "sandwich", "orange", "broccoli", "carrot", "hot_dog", "pizza", "donut", "cake", "chair",
                    "couch",
                    "potted plant", "bed", "mirror", "dining table", "window", "desk", "toilet", "door", "tv",
                    "laptop",
                    "mouse", "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink",
                    "refrigerator",
                    "blender", "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush",
                    "hair brush"]

transform = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor()
])


def save_threshold_item(pred, threshold):
    """
    保留置信度>thresh_hold的pred

    :param pred: 模型输出
    :param threshold: 阈值
    :return: 返回截取后的pred
    """
    threshold_index = 0
    for i in range(len(pred)):
        for j, score in enumerate(pred[i]['scores']):
            if score <= threshold:
                threshold_index = j
                break
        pred[i]['boxes'] = pred[i]['boxes'][:threshold_index]
        pred[i]['labels'] = pred[i]['labels'][:threshold_index]
        pred[i]['scores'] = pred[i]['scores'][:threshold_index]

    return pred


def plot(pred, images):
    """
    在图像上绘制box，label，confidence
    :param pred: 模型输出
    :param image: opencv读取的图像，可以用列表装入多张图像
    :return:
    """

    for i, image in enumerate(images):

        boxes = pred[i]['boxes']
        labels = pred[i]['labels']
        scores = pred[i]['scores']
        for i, item in enumerate(zip(boxes, labels, scores)):
            xmin = int(item[0][0])
            ymin = int(item[0][1])
            xmax = int(item[0][2])
            ymax = int(item[0][3])
            # 绘制box
            cv.rectangle(img=image, pt1=[xmin, ymin], pt2=[xmax, ymax], color=[255, 0, 0], thickness=2)
            # 取出置信度
            confidence = item[2].cpu().detach().clone().numpy()
            confidence = round(float(confidence), 2)
            text = coco_labels_name[int(item[1].cpu().numpy())] + " " + str(confidence)
            font = cv.FONT_HERSHEY_SIMPLEX
            cv.putText(image, text, (xmin, ymin), font, 0.8, (0, 0, 255), 1, cv.LINE_AA)
    return images


def crop_image_to_graph(pred, frame, img):
    for i, image in enumerate(frame):
        boxes = pred[i]['boxes']
        labels = pred[i]['labels']
        scores = pred[i]['scores']
        len = 0
        arr = []
        G = nx.Graph()
        for i, item in enumerate(zip(boxes, labels, scores)):
            xmin = int(item[0][0])
            ymin = int(item[0][1])
            xmax = int(item[0][2])
            ymax = int(item[0][3])
            # 裁剪指定区域
            crop_img = img.crop((xmin, ymin, xmax, ymax))
            # 缩放到相同大小
            resize_img = crop_img.resize((50, 50))
            # 转换为向量表示
            vec = np.array(resize_img).flatten()
            # vec = tuple(vec)
            arr.append(vec)
            # 坑：G的图，左下角为 (0, 0)，CV的图，左上角为 (0, 0)
            G.add_node(len, value=vec, pos=(float((xmin + xmax) / 2), float((1000 - ymin + 1000 - ymax) / 2)),
                       label=coco_labels_name[int(item[1].cpu().numpy())])
            len = len + 1
        # 这段添加边的权重的代码有问题，但是问题是真的需要权重嘛
        # 遍历所有节点
        for u in G.nodes():
            for v in G.nodes():
                if u != v:
                    # 获取节点 u 和节点 v 的值
                    value_u = G.nodes[u]['value']
                    value_v = G.nodes[v]['value']

                    # 计算欧氏距离
                    distance = np.linalg.norm(value_u - value_v)

                    # 添加或更新节点 u 和节点 v 之间的边及其属性
                    if G.has_edge(u, v):
                        G[u][v]['weight'] = distance
                    else:
                        G.add_edge(u, v, weight=distance)

        # # 输出更新后的权重
        # for u, v, data in G.edges(data=True):
        #     weight = data['weight']
        #     print(f"Edge ({u}, {v}): {weight}")
        return G


def image2graph(filename):
    """
    实现将图像数据变为图结构
    :param filename: 图像数据地址
    :return: 图结构G和原始图像数据img
    """
    model.eval()

    # 读取照片
    image = Image.open(filename).convert("RGB")

    # 将照片转换为ndarray形式
    frame = np.array(image)

    tensor_frame = transform(frame)
    # 将[C, H, W]转化为[1, C, H, W]
    depth = frame.shape[2]
    width = frame.shape[0]
    height = frame.shape[1]
    tensor_frame = torch.reshape(tensor_frame, (1, depth, width, height))
    tensor_frame = tensor_frame.to(device)
    # 得到模型输出
    pred = model(tensor_frame)
    # 对pred进行截断，保留confidence > threshold的部分
    threshold = 0.8
    pred = save_threshold_item(pred, threshold)
    # 在图像上绘制box，label，confidence
    [frame] = plot(pred, [frame])
    # 将ndarray转换为图像
    img = Image.fromarray(frame)
    # 转换为向量表示
    G = crop_image_to_graph(pred, [frame], img)
    return G, img

    # # 可视化图结构
    # img.show()
    # plt.rcParams['font.sans-serif'] = ['KaiTi', 'SimHei', 'FangSong']  # 汉字字体,优先使用楷体，如果找不到楷体，则使用黑体
    # plt.rcParams['font.size'] = 12  # 字体大小
    # plt.rcParams['axes.unicode_minus'] = False  # 正常显示负号
    # # # 提取节点坐标信息
    # node_positions = nx.get_node_attributes(G, 'pos')
    # edges = [(u, v) for u, v, d in G.edges(data=True) if d['weight'] >= 5]  # 控制边的权重大于阈值的进行显示
    # nx.draw(G, pos=node_positions, labels={n: str(d['label']) for n, d in G.nodes(data=True)})
    # # 显示图形
    # plt.show()
