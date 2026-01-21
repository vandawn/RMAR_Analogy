import torch
import torchvision.models as models
from torchvision import transforms
from PIL import Image
import os
import pickle
from tqdm import tqdm
import collections
import glob
import urllib.parse # 导入URL编码库

def setup_model():
    """
    加载并配置预训练的VGG16模型，移除最后的全连接层。
    """
    # 加载预训练的VGG16模型
    vgg16 = models.vgg16(pretrained=True)
    
    # 按照原始脚本的结构重命名特征提取层
    vgg16.features = torch.nn.Sequential(collections.OrderedDict(zip(
        ['conv1_1', 'relu1_1', 'conv1_2', 'relu1_2', 'pool1', 'conv2_1', 'relu2_1', 'conv2_2', 'relu2_2', 'pool2', 
         'conv3_1', 'relu3_1', 'conv3_2', 'relu3_2', 'conv3_3', 'relu3_3', 'pool3', 'conv4_1', 'relu4_1', 
         'conv4_2', 'relu4_2', 'conv4_3', 'relu4_3', 'pool4', 'conv5_1', 'relu5_1', 'conv5_2', 'relu5_2', 
         'conv5_3', 'relu5_3', 'pool5'], vgg16.features
    )))
    
    # 修改分类器，只保留到fc7层，以获取4096维的特征
    vgg16.classifier = torch.nn.Sequential(collections.OrderedDict(zip(
        ['fc6', 'relu6', 'drop6', 'fc7'], vgg16.classifier
    )))
    
    print("VGG16模型结构配置完成。")
    print(vgg16.classifier)
    
    return vgg16

def get_image_transform():
    """
    定义图像预处理的转换流程。
    """
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

def main():
    """
    主流程：遍历实体，为其生成并保存平均图像嵌入。
    """
    # --- 1. 配置 ---
    BASE_PATH = "/home/rwan551/code/MKG_Analogy-main"
    
    # 输入文件：包含实体字符串和对应数字ID的列表
    ENTITY_ID_FILE = os.path.join(BASE_PATH, "M-KGE/IKRL_TransAE/data/MCNet/entity2id.txt")
    
    # 输入目录：存放所有实体图片文件夹的根目录
    # 注意：这里需要指向您实际存放所有 'part' 文件夹的父目录
    IMAGE_BASE_DIR = os.path.join(BASE_PATH, "MarT/dataset/MCNetAnalogy/images") 
    
    # 输出目录：存放生成的 embedding.pkl 文件
    OUTPUT_EMBED_DIR = os.path.join(BASE_PATH, "M-KGE/IKRL_TransAE/data/MCNet/visual_embeds")

    # 自动选择设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"将使用设备: {device}")

    # --- 2. 初始化模型和图像转换器 ---
    model = setup_model()
    model.to(device)
    model.eval() # 设置为评估模式
    
    transformation = get_image_transform()
    
    # 确保输出目录存在
    os.makedirs(OUTPUT_EMBED_DIR, exist_ok=True)

    # --- 3. 读取实体列表 ---
    try:
        with open(ENTITY_ID_FILE, 'r', encoding='utf-8') as f:
            entity_lines = f.readlines()
    except FileNotFoundError:
        print(f"错误: 实体ID文件未找到 -> {ENTITY_ID_FILE}")
        return

    # --- 4. 遍历每个实体并生成嵌入 ---
    print(f"\n开始为 {len(entity_lines)} 个实体生成图像嵌入...")
    
    for line in tqdm(entity_lines, desc="处理实体中"):
        parts = line.strip().split('\t')
        if len(parts) != 2:
            continue
            
        entity_str, entity_id_num = parts[0], parts[1]
        
        # --- 核心修改：对实体字符串进行URL编码，以匹配文件夹名称 ---
        # safe='' 确保斜杠'/'也被编码
        encoded_entity_str = urllib.parse.quote(entity_str, safe='')
        
        # 拼接图片文件夹路径
        # 由于图片分在不同part文件夹下，我们需要 glob 来查找
        image_folder_pattern = os.path.join(IMAGE_BASE_DIR, "part*", encoded_entity_str)
        found_folders = glob.glob(image_folder_pattern)

        if not found_folders:
            # print(f"信息: 实体 '{entity_str}' (ID: {entity_id_num}) 没有找到对应的图片文件夹。")
            continue
        
        # 理论上只有一个匹配项，我们用第一个
        image_folder = found_folders[0]
        image_paths = glob.glob(os.path.join(image_folder, "*.*"))

        if not image_paths:
            # print(f"信息: 文件夹 '{image_folder}' 为空。")
            continue

        # 加载并转换该实体的所有图片
        images_tensor_list = []
        for img_path in image_paths:
            try:
                img = Image.open(img_path).convert("RGB")
                img_tensor = transformation(img)
                images_tensor_list.append(img_tensor)
            except Exception as e:
                print(f"警告: 无法打开或处理图片 {img_path}。错误: {e}")
        
        if not images_tensor_list:
            continue
            
        # 将图片列表堆叠成一个张量
        image_batch = torch.stack(images_tensor_list).to(device)
        
        # --- 5. 计算并保存嵌入 ---
        with torch.no_grad(): # 在评估模式下，不需要计算梯度
            embeddings = model(image_batch)
            # 计算所有图片嵌入的平均值
            avg_embedding = embeddings.mean(dim=0)

        # 将最终的嵌入向量移回CPU以便保存
        avg_embedding_cpu = avg_embedding.cpu()

        # 定义输出路径，使用数字ID作为文件夹名
        output_entity_dir = os.path.join(OUTPUT_EMBED_DIR, entity_id_num)
        os.makedirs(output_entity_dir, exist_ok=True)
        output_file_path = os.path.join(output_entity_dir, "avg_embedding.pkl")
        
        with open(output_file_path, "wb") as f:
            pickle.dump(avg_embedding_cpu, f)

    print("\n所有实体的图像嵌入生成完毕！")
    print(f"文件保存在: {OUTPUT_EMBED_DIR}")

if __name__ == "__main__":
    main()