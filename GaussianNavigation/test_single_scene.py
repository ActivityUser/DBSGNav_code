import os
import json
import sys
sys.path.append('.')
from gaussian_utils.build_instance_clip_vectors import CLIPEncoder, _resolve_rgb_path

scene_path = "configs/data/habitat_val/4ok3usBNeis"
ir_path = os.path.join(scene_path, "instance_retrieval.json")

print("加载 instance_retrieval.json...")
with open(ir_path, 'r') as f:
    data = json.load(f)

print("初始化 CLIP 编码器...")
encoder = CLIPEncoder(model_name="ViT-B-32", device="cuda")

COCO_IDS = [56, 57, 58, 59, 61, 62]
result = {}

for coco_id in COCO_IDS:
    key = str(coco_id)
    if key not in data:
        result[key] = []
        continue
    
    entries = data[key]
    vectors = []
    print(f"\n处理类别 {key}, {len(entries)} 个实例")
    
    for idx, entry in enumerate(entries):
        try:
            if len(entry) >= 2:
                frame_id = entry[1]
                # 直接构建路径
                rgb_path = os.path.join(scene_path, "rgb", f"frame_{frame_id}.png")
                
                if os.path.exists(rgb_path):
                    vec = encoder.encode_image(rgb_path)
                    if vec is None:
                        print(f"  [{idx}] 编码失败: {frame_id}")
                        vec = [0.0] * 512
                else:
                    print(f"  [{idx}] 文件不存在: {rgb_path}")
                    vec = [0.0] * 512
            else:
                vec = [0.0] * 512
        except Exception as e:
            print(f"  [{idx}] 错误: {e}")
            vec = [0.0] * 512
        
        vectors.append(vec)
        if idx < 3 and any(abs(v) > 1e-6 for v in vec):
            print(f"  [{idx}] 成功: {frame_id}, 非零值")
    
    result[key] = vectors

# 保存结果
out_path = os.path.join(scene_path, "instance_clip_vectors_new.json")
with open(out_path, 'w') as f:
    json.dump(result, f)

print(f"\n结果已保存到: {out_path}")

# 检查是否有非零向量
total_vectors = sum(len(v) for v in result.values())
non_zero = 0
for key in result:
    for vec in result[key]:
        if any(abs(v) > 1e-6 for v in vec):
            non_zero += 1
print(f"总向量数: {total_vectors}, 非零向量: {non_zero}")
