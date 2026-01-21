# 读取实体列表
with open(entity2text_path, 'r', encoding='utf-8') as f:
    all_entities = [line.strip().split('\t')[0] for line in f]

entity2visual = []

for ent_id in tqdm(all_entities):
    path = os.path.join('dataset/MARS/images', ent_id)
    if not os.path.isdir(path):
        print(f"[WARN] No image for {ent_id}, using zero feature")
        entity2visual.append(torch.zeros(3, 384, 384))  # 或者其他 dummy vector
        continue

    files = os.listdir(path)
    if not files:
        print(f"[WARN] Empty folder for {ent_id}, using zero feature")
        entity2visual.append(torch.zeros(3, 384, 384))
        continue

    file = random.choice(files)
    image = Image.open(os.path.join(path, file)).convert('RGB').resize((384, 384))
    pixel_values = processor(images=image, text="test", return_tensors='pt')['pixel_values'].squeeze(0)
    entity2visual.append(pixel_values)

torch.save(torch.stack(entity2visual), 'dataset/MARS_OOD/entity_image_features_vilt.pth')
